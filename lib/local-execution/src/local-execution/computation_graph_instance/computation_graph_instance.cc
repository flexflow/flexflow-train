#include "local-execution/computation_graph_instance/computation_graph_instance.h"
#include "kernels/allocation.h"
#include "local-execution/local_task_registry.h"
#include "local-execution/task_execution.h"
#include "local-execution/tensor_allocation.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_cg.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/update_insertion.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/all_of.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_values_strict.h"
#include "utils/exception.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/graph/instances/unordered_set_labelled_open_dataflow_graph.h"
#include "utils/graph/instances/unordered_set_labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/labelled_open_kwarg_dataflow_graph/labelled_open_kwarg_dataflow_graph.h"
#include "utils/graph/node/algorithms.h"
#include "utils/many_to_one/many_to_one.h"
#include "utils/optional.h"
#include <optional>
#include <unordered_map>

namespace FlexFlow {

bool no_nodes_are_initialized(DynamicOpenDataflowGraph const &g) {
  return all_are_true(
      transform(get_dynamic_nodes(g), [](DynamicNodeAttrs const &n) -> bool {
        return !n.per_device_op_state.has_value();
      }));
}

bool all_nodes_are_initialized(DynamicOpenDataflowGraph const &g) {
  return all_are_true(
      transform(get_dynamic_nodes(g), [](DynamicNodeAttrs const &n) -> bool {
        return n.per_device_op_state.has_value();
      }));
}

ComputationGraphInstance::ComputationGraphInstance(DynamicOpenDataflowGraph dg,
                                                   Allocator &alloc)
    : dataflow_graph(dg), allocator(alloc) {}

DynamicNodeInvocation
    initialize_node(DynamicNodeInvocation const &i,
                    Allocator &allocator,
                    ProfilingSettings const &profiling_settings,
                    device_handle_t const &device_handle,
                    DeviceType kernel_device_type,
                    FFIterationConfig const &iteration_config,
                    size_t device_idx) {
  // Get op
  ComputationGraphOpAttrs op_attrs =
      assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
          assert_unwrap(i.node_attrs.op_attrs)));

  // Prepare arguments
  TaskArgumentAccessor arg_accessor =
      make_task_argument_accessor_for_invocation(
          /*invocation=*/i,
          /*profiling_settings=*/profiling_settings,
          /*ff_handle=*/device_handle,
          /*kernel_device_type=*/kernel_device_type,
          /*op_attrs=*/assert_unwrap(i.node_attrs.op_attrs),
          /*loss_attrs=*/std::nullopt,
          /*per_device_op_state=*/std::nullopt,
          /*iteration_config=*/iteration_config,
          /*optimizer_attrs=*/std::nullopt,
          /*device_idx=*/device_idx);

  // Run task init
  std::optional<DeviceSpecificPerDeviceOpState> per_device_op_state =
      call_init_task_impl(op_attrs, arg_accessor);

  DynamicNodeAttrs node_attrs{
      /*task_type=*/i.node_attrs.task_type,
      /*device_coord=*/i.node_attrs.device_coord,
      /*mapping=*/i.node_attrs.mapping,
      /*op_attrs=*/i.node_attrs.op_attrs,
      /*layer_guid=*/i.node_attrs.layer_guid,
      /*per_device_op_state=*/per_device_op_state,
  };
  return DynamicNodeInvocation{
      /*inputs=*/
      i.inputs,
      /*node_attrs=*/
      node_attrs,
      /*outputs=*/
      i.outputs,
  };
}

ComputationGraphInstance create_computation_graph_instance(
    ComputationGraph const &cg,
    OptimizerAttrs const &optimizer,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &input_tensors,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    DeviceType kernel_device_type,
    FFIterationConfig const &iteration_config,
    size_t device_idx) {
  DynamicOpenDataflowGraph dg = make_dynamic_open_dataflow_graph_from_cg(cg);
  dg = perform_pass_expansion(dg);
  dg = perform_update_insertion(dg, optimizer);
  dg = perform_tensor_allocation(dg, input_tensors, allocator);

  // Initialize all operators and save the per-device op state
  ASSERT(no_nodes_are_initialized(dg));
  dg = transform_dynamic_invocation_set(
      dg, [&](DynamicNodeInvocation const &invocation) {
        return initialize_node(invocation,
                               allocator,
                               profiling_settings,
                               device_handle,
                               kernel_device_type,
                               iteration_config,
                               device_idx);
      });
  ASSERT(all_nodes_are_initialized(dg));

  return ComputationGraphInstance{dg, allocator};
}

std::pair<LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                         DynamicValueAttrs,
                                         int,
                                         DynamicTensorSlot>,
          std::unordered_map<Node, DynamicNodeInvocation>>
    labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph_with_map(
        DynamicOpenDataflowGraph const &g) {

  std::unordered_set<DynamicValueAttrs> all_values =
      unordered_set_of(get_dynamic_values(g));

  ManyToOne<DynamicValueAttrs, DynamicNodeInvocation> value_to_producer;
  for (DynamicNodeInvocation const &invocation :
       get_dynamic_invocation_set(g)) {
    for (DynamicValueAttrs const &output : values(invocation.outputs)) {
      value_to_producer.insert({output, invocation});
    }
  }

  std::unordered_set<DynamicValueAttrs> graph_inputs =
      filter(all_values, [&](DynamicValueAttrs const &v) -> bool {
        return !value_to_producer.contains_l(v);
      });

  LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                 DynamicValueAttrs,
                                 int,
                                 DynamicTensorSlot>
      result = LabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                              DynamicValueAttrs,
                                              int,
                                              DynamicTensorSlot>::
          create<
              UnorderedSetLabelledOpenKwargDataflowGraph<DynamicNodeAttrs,
                                                         DynamicValueAttrs,
                                                         int,
                                                         DynamicTensorSlot>>();

  std::unordered_map<Node, DynamicNodeInvocation> node_map;
  bidict<OpenKwargDataflowValue<int, DynamicTensorSlot>, DynamicValueAttrs>
      value_map;

  for (auto const &kv : enumerate(graph_inputs)) {
    int input_idx = kv.first.unwrap_nonnegative();
    DynamicValueAttrs graph_input = kv.second;
    KwargDataflowGraphInput<int> added =
        result.add_input(input_idx, graph_input);
    value_map.equate(OpenKwargDataflowValue<int, DynamicTensorSlot>{added},
                     graph_input);
  }

  auto inputs_have_been_added =
      [&](DynamicNodeInvocation const &invocation) -> bool {
    return all_of(values(invocation.inputs),
                  [&](DynamicValueAttrs const &input) -> bool {
                    return value_map.contains_r(input);
                  });
  };

  std::unordered_set<DynamicNodeInvocation> to_add = g.invocations;

  auto add_invocation_to_graph =
      [&](DynamicNodeInvocation const &invocation) -> void {
    KwargNodeAddedResult<DynamicTensorSlot> added = result.add_node(
        invocation.node_attrs,
        map_values(invocation.inputs,
                   [&](DynamicValueAttrs const &input)
                       -> OpenKwargDataflowValue<int, DynamicTensorSlot> {
                     return value_map.at_r(input);
                   }),
        invocation.outputs);
    node_map.insert(std::pair{added.node, invocation});

    for (auto const &[k, v] :
         zip_values_strict(invocation.outputs, added.outputs)) {
      DynamicValueAttrs invocation_output = v.first;
      KwargDataflowOutput<DynamicTensorSlot> graph_output = v.second;
      value_map.equate(
          OpenKwargDataflowValue<int, DynamicTensorSlot>{graph_output},
          invocation_output);
    }

    to_add.erase(invocation);
  };

  auto add_next_invocation_to_graph = [&]() {
    for (DynamicNodeInvocation const &invocation : to_add) {
      if (inputs_have_been_added(invocation)) {
        add_invocation_to_graph(invocation);
        return;
      }
    }

    PANIC("Failed to add any invocations in to_add", to_add);
  };

  while (to_add.size() > 0) {
    add_next_invocation_to_graph();
  }

  return std::pair{result, node_map};
}

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_all_passes_for_computation_graph_instance(
        ComputationGraphInstance const &instance) {
  std::pair<LabelledOpenKwargDataflowGraphView<DynamicNodeAttrs,
                                               DynamicValueAttrs,
                                               int,
                                               DynamicTensorSlot>,
            std::unordered_map<Node, DynamicNodeInvocation>>
      dataflow_graph_and_map =
          labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph_with_map(
              instance.get_dynamic_dataflow_graph());
  LabelledOpenKwargDataflowGraphView<DynamicNodeAttrs,
                                     DynamicValueAttrs,
                                     int,
                                     DynamicTensorSlot>
      dataflow_graph = dataflow_graph_and_map.first;
  std::unordered_map<Node, DynamicNodeInvocation> node_map =
      dataflow_graph_and_map.second;
  std::vector<Node> nodes = get_topological_ordering(dataflow_graph);
  for (Node const &node : nodes) {
    DynamicNodeInvocation invocation = node_map.at(node);
  }
  NOT_IMPLEMENTED();
}

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

void perform_update_pass_for_computation_graph_instance(
    ComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
