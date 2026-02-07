#include "local-execution/computation_graph_instance/computation_graph_instance.h"
#include "local-execution/device_state_initialization.h"
#include "local-execution/task_execution.h"
#include "local-execution/tensor_allocation.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_guid_t.dtg.h"
#include "task-spec/dynamic_graph/dynamic_value_attrs.dtg.h"
#include "task-spec/dynamic_graph/loss_insertion.h"
#include "task-spec/dynamic_graph/make_dynamic_open_dataflow_graph_from_cg.h"
#include "task-spec/dynamic_graph/pass_expansion.h"
#include "task-spec/dynamic_graph/update_insertion.h"
#include "task-spec/per_device_op_state.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_map_from_pairs.h"
#include "utils/graph/digraph/algorithms/get_topological_ordering.h"
#include "utils/optional.h"

namespace FlexFlow {

ComputationGraphInstance::ComputationGraphInstance(
    Allocator &allocator,
    std::vector<DynamicNodeInvocation> const &execution_order,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<GenericTensorAccessorW> logit_grad_tensor)
    : allocator(allocator), execution_order(execution_order),
      optimizer_attrs(optimizer_attrs), loss_attrs(loss_attrs),
      logit_grad_tensor(logit_grad_tensor) {}

Allocator &ComputationGraphInstance::get_allocator() const {
  return this->allocator;
}
std::vector<DynamicNodeInvocation> const &
    ComputationGraphInstance::get_execution_order() const {
  return this->execution_order;
}
OptimizerAttrs const &ComputationGraphInstance::get_optimizer_attrs() const {
  return this->optimizer_attrs;
}
void ComputationGraphInstance::update_optimizer_attrs_for_next_iter() {
  this->optimizer_attrs =
      get_optimizer_attrs_for_next_iter(this->optimizer_attrs);
}
std::optional<LossAttrs> const &
    ComputationGraphInstance::get_loss_attrs() const {
  return this->loss_attrs;
}
std::optional<GenericTensorAccessorR>
    ComputationGraphInstance::get_loss_tensor_accessor() const {
  return this->logit_grad_tensor;
}

static GenericTensorAccessorW
    get_loss_tensor_accessor(DynamicOpenDataflowGraph const &dg,
                             DynamicValueAttrs const &value) {
  return assert_unwrap(assert_unwrap(find_output_tensor(
                                         dg, value.tensor_guid, value.role))
                           .second.accessor)
      .get<GenericTensorAccessorW>();
}

ComputationGraphInstance create_computation_graph_instance(
    ComputationGraph const &cg,
    OptimizerAttrs const &optimizer_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<GenericTensorAccessorR> label_tensor,
    std::optional<tensor_guid_t> logit_tensor,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &input_tensors,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    FFIterationConfig const &iteration_config,
    device_id_t device_idx) {
  DynamicOpenDataflowGraph dg = make_dynamic_open_dataflow_graph_from_cg(cg);
  dg = perform_pass_expansion(dg);

  std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> inputs =
      input_tensors;
  std::optional<DynamicValueAttrs> logit_grad_value;
  if (loss_attrs) {
    auto [dg2, label_v, logit_grad_v] = perform_loss_insertion(
        dg,
        assert_unwrap(loss_attrs),
        dynamic_tensor_guid_t{assert_unwrap(logit_tensor)});
    dg = dg2;
    logit_grad_value = logit_grad_v;
    inputs.insert(std::pair{label_v, assert_unwrap(label_tensor)});
  }

  dg = perform_update_insertion(dg, optimizer_attrs);
  dg = perform_tensor_allocation(dg, inputs, allocator);

  std::optional<GenericTensorAccessorW> logit_grad_tensor =
      transform(logit_grad_value, [&](DynamicValueAttrs const &lgv) {
        return get_loss_tensor_accessor(dg, lgv);
      });

  dg = perform_device_state_initialization(dg,
                                           allocator,
                                           profiling_settings,
                                           device_handle,
                                           iteration_config,
                                           optimizer_attrs,
                                           device_idx);

  // Compute the topological ordering of the graph
  auto [kwarg_graph, node_map] =
      labelled_open_kwarg_dataflow_graph_from_dynamic_open_dataflow_graph(dg);
  std::vector<Node> node_topo_order = get_topological_ordering(kwarg_graph);
  std::vector<DynamicNodeInvocation> invocation_topo_order = transform(
      node_topo_order, [&](Node node) { return node_map.at_l(node); });

  return ComputationGraphInstance{allocator,
                                  invocation_topo_order,
                                  optimizer_attrs,
                                  loss_attrs,
                                  logit_grad_tensor};
}

static std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
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
            transform(invocation.node_attrs.per_device_op_state,
                      [&](DeviceSpecificPerDeviceOpState const &op_state) {
                        return get_device_state_from_device_specific(
                            op_state, device_idx);
                      }),
            /*iteration_config=*/iteration_config,
            /*optimizer_attrs=*/optimizer_attrs,
            /*device_idx=*/device_idx);
        return std::pair{invocation.node_attrs.layer_guid, timing};
      }));
}

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_all_passes_for_computation_graph_instance(
        ComputationGraphInstance &instance,
        ProfilingSettings const &profiling_settings,
        device_handle_t const &ff_handle,
        FFIterationConfig iteration_config,
        device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> const &execution_order =
      instance.get_execution_order();
  std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
      result = execute_dynamic_node_invocation_set(
          /*invocations=*/execution_order,
          /*allocator=*/instance.get_allocator(),
          /*optimizer_attrs=*/instance.get_optimizer_attrs(),
          /*profiling_settings=*/profiling_settings,
          /*ff_handle=*/ff_handle,
          /*loss_attrs=*/instance.get_loss_attrs(),
          /*iteration_config=*/iteration_config,
          /*device_idx=*/device_idx);
  instance.update_optimizer_attrs_for_next_iter();
  return result;
}

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &instance,
        ProfilingSettings const &profiling_settings,
        device_handle_t const &ff_handle,
        FFIterationConfig iteration_config,
        device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> const &execution_order =
      filter(instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::FWD;
             });

  return execute_dynamic_node_invocation_set(
      /*invocations=*/execution_order,
      /*allocator=*/instance.get_allocator(),
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*loss_attrs=*/instance.get_loss_attrs(),
      /*iteration_config=*/iteration_config,
      /*device_idx=*/device_idx);
}

std::unordered_map<dynamic_layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        ComputationGraphInstance const &instance,
        ProfilingSettings const &profiling_settings,
        device_handle_t const &ff_handle,
        FFIterationConfig iteration_config,
        device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> const &execution_order =
      filter(instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::BWD;
             });

  return execute_dynamic_node_invocation_set(
      /*invocations=*/execution_order,
      /*allocator=*/instance.get_allocator(),
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*loss_attrs=*/instance.get_loss_attrs(),
      /*iteration_config=*/iteration_config,
      /*device_idx=*/device_idx);
}

void perform_update_pass_for_computation_graph_instance(
    ComputationGraphInstance &instance,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    FFIterationConfig iteration_config,
    device_id_t device_idx) {
  std::vector<DynamicNodeInvocation> const &execution_order =
      filter(instance.get_execution_order(),
             [](DynamicNodeInvocation const &invocation) {
               DynamicTaskType task_type =
                   assert_unwrap(invocation.node_attrs.task_type);
               return task_type == DynamicTaskType::UPD;
             });

  execute_dynamic_node_invocation_set(
      /*invocations=*/execution_order,
      /*allocator=*/instance.get_allocator(),
      /*optimizer_attrs=*/instance.get_optimizer_attrs(),
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*loss_attrs=*/instance.get_loss_attrs(),
      /*iteration_config=*/iteration_config,
      /*device_idx=*/device_idx);
  instance.update_optimizer_attrs_for_next_iter();
}

} // namespace FlexFlow
