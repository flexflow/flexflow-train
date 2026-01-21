#include "local-execution/computation_graph_instance/computation_graph_instance.h"
#include "kernels/allocation.h"
#include "local-execution/local_task_registry.h"
#include "local-execution/task_execution.h"
#include "local-execution/tensor_allocation.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_cg.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/update_insertion.h"
#include "task-spec/per_device_op_state.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_map_from_pairs.h"
#include "utils/exception.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/optional.h"
#include <cassert>
#include <optional>
#include <unordered_map>

namespace FlexFlow {

ComputationGraphInstance::ComputationGraphInstance(
    DynamicOpenDataflowGraph dataflow_graph,
    Allocator &allocator,
    std::vector<DynamicNodeInvocation> const &topological_ordering,
    OptimizerAttrs const &optimizer_attrs)
    : dataflow_graph(dataflow_graph), allocator(allocator),
      topological_ordering(topological_ordering),
      optimizer_attrs(optimizer_attrs) {}

DynamicOpenDataflowGraph const &
    ComputationGraphInstance::get_dynamic_dataflow_graph() const {
  return this->dataflow_graph;
}
Allocator &ComputationGraphInstance::get_allocator() const {
  return this->allocator;
}
std::vector<DynamicNodeInvocation> const &
    ComputationGraphInstance::get_topological_ordering() const {
  return this->topological_ordering;
}
OptimizerAttrs const &ComputationGraphInstance::get_optimizer_attrs() const {
  return this->optimizer_attrs;
}

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

DynamicNodeInvocation
    initialize_node(DynamicNodeInvocation const &i,
                    Allocator &allocator,
                    ProfilingSettings const &profiling_settings,
                    device_handle_t const &device_handle,
                    FFIterationConfig const &iteration_config,
                    device_id_t device_idx) {
  // Get op
  ComputationGraphOpAttrs op_attrs =
      assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
          assert_unwrap(i.node_attrs.op_attrs)));

  // Prepare arguments
  TaskArgumentAccessor arg_accessor =
      make_task_argument_accessor_for_invocation(
          /*invocation=*/i,
          /*allocator=*/allocator,
          /*profiling_settings=*/profiling_settings,
          /*ff_handle=*/device_handle,
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
    OptimizerAttrs const &optimizer_attrs,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &input_tensors,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    FFIterationConfig const &iteration_config,
    device_id_t device_idx) {
  DynamicOpenDataflowGraph dg = make_dynamic_open_dataflow_graph_from_cg(cg);
  dg = perform_pass_expansion(dg);
  dg = perform_update_insertion(dg, optimizer_attrs);
  dg = perform_tensor_allocation(dg, input_tensors, allocator);

  // Initialize all operators and save the per-device op state
  ASSERT(no_nodes_are_initialized(dg));
  dg = transform_dynamic_invocation_set(
      dg, [&](DynamicNodeInvocation const &invocation) {
        return initialize_node(invocation,
                               allocator,
                               profiling_settings,
                               device_handle,
                               iteration_config,
                               device_idx);
      });
  ASSERT(all_nodes_are_initialized(dg));

  // Compute the topological ordering of the graph
  auto [kwarg_graph, node_map] =
      labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(dg);
  std::vector<Node> node_topo_order = get_topological_ordering(kwarg_graph);
  std::vector<DynamicNodeInvocation> invocation_topo_order = transform(
      node_topo_order, [&](Node node) { return node_map.at_l(node); });

  return ComputationGraphInstance{
      dg, allocator, invocation_topo_order, optimizer_attrs};
}

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    execute_dynamic_node_invocation_set(
        std::vector<DynamicNodeInvocation> const &invocations,
        Allocator &allocator,
        OptimizerAttrs const &optimizer_attrs,
        ProfilingSettings const &profiling_settings,
        device_handle_t const &ff_handle,
        std::optional<LossAttrs> const &loss_attrs,
        FFIterationConfig iteration_config,
        device_id_t device_idx) {
  return unordered_map_from_pairs(
      transform(invocations, [&](DynamicNodeInvocation const &invocation) {
        std::optional<milliseconds_t> timing = execute_dynamic_node_invocation(
            /*invocation=*/invocation,
            /*allocator=*/allocator,
            /*profiling_settings=*/profiling_settings,
            /*ff_handle=*/ff_handle,
            /*loss_attrs=*/loss_attrs,
            /*per_device_op_state=*/
            get_device_state_from_device_specific(
                assert_unwrap(invocation.node_attrs.per_device_op_state),
                device_idx),
            /*iteration_config=*/iteration_config,
            /*optimizer_attrs=*/optimizer_attrs,
            /*device_idx=*/device_idx);
        return std::pair{invocation.node_attrs.layer_guid, timing};
      }));
}

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_all_passes_for_computation_graph_instance(
        ComputationGraphInstance const &instance,
        ProfilingSettings const &profiling_settings,
        device_handle_t const &ff_handle,
        std::optional<LossAttrs> const &loss_attrs,
        FFIterationConfig iteration_config,
        device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> const &topo_order =
      instance.get_topological_ordering();
  return execute_dynamic_node_invocation_set(
      /*invocations=*/topo_order,
      /*allocator=*/instance.get_allocator(),
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*loss_attrs=*/loss_attrs,
      /*iteration_config=*/iteration_config,
      /*device_idx=*/device_idx);
}

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &instance,
        ProfilingSettings const &profiling_settings,
        device_handle_t const &ff_handle,
        std::optional<LossAttrs> const &loss_attrs,
        FFIterationConfig iteration_config,
        device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> const &topo_order =
      filter(instance.get_topological_ordering(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::FWD;
             });

  return execute_dynamic_node_invocation_set(
      /*invocations=*/topo_order,
      /*allocator=*/instance.get_allocator(),
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*loss_attrs=*/loss_attrs,
      /*iteration_config=*/iteration_config,
      /*device_idx=*/device_idx);
}

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &instance,
        ProfilingSettings const &profiling_settings,
        device_handle_t const &ff_handle,
        std::optional<LossAttrs> const &loss_attrs,
        FFIterationConfig iteration_config,
        device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> const &topo_order =
      filter(instance.get_topological_ordering(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::BWD;
             });

  return execute_dynamic_node_invocation_set(
      /*invocations=*/topo_order,
      /*allocator=*/instance.get_allocator(),
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*loss_attrs=*/loss_attrs,
      /*iteration_config=*/iteration_config,
      /*device_idx=*/device_idx);
}

void perform_update_pass_for_computation_graph_instance(
    ComputationGraphInstance const &instance,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<LossAttrs> const &loss_attrs,
    FFIterationConfig iteration_config,
    device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> const &topo_order =
      filter(instance.get_topological_ordering(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::UPD;
             });

  execute_dynamic_node_invocation_set(
      /*invocations=*/topo_order,
      /*allocator=*/instance.get_allocator(),
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*loss_attrs=*/loss_attrs,
      /*iteration_config=*/iteration_config,
      /*device_idx=*/device_idx);
}

} // namespace FlexFlow
