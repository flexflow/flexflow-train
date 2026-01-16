#include "local-execution/computation_graph_instance/initialized_computation_graph_instance.h"
#include "kernels/allocation.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/local_task_registry.h"
#include "local-execution/task_execution.h"
#include "local-execution/tensor_allocation.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "task-spec/dynamic_graph/dynamic_open_dataflow_graph.h"
#include "task-spec/dynamic_graph/dynamic_tensor_accessor.dtg.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"
#include "task-spec/task_id_with_noop_default_t.dtg.h"
#include "task-spec/task_id_with_noop_default_t.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/transform.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include <alloca.h>
#include <cassert>
#include <cmath>
#include <optional>
#include <vector>

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

InitializedComputationGraphInstance::InitializedComputationGraphInstance(
    DynamicOpenDataflowGraph dg, Allocator &alloc, LocalTaskRegistry &registry)
    : initialized_dataflow_graph(dg), allocator(alloc),
      task_registry(registry) {}

DynamicNodeInvocation
    initialize_node(DynamicNodeInvocation const &i,
                    Allocator &allocator,
                    LocalTaskRegistry &task_registry,
                    ProfilingSettings const &profiling_settings,
                    device_handle_t const &device_handle,
                    DeviceType kernel_device_type,
                    FFIterationConfig const &iteration_config,
                    size_t device_idx) {
  // Get task
  task_id_with_noop_default_t registered_task = get_init_task_id_for_op_attrs(
      assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
          assert_unwrap(i.node_attrs.op_attrs))));

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
      call_init_task_impl(task_registry, registered_task, arg_accessor);

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

InitializedComputationGraphInstance initialize_computation_graph_instance(
    ComputationGraphInstance const &instance,
    std::unordered_map<DynamicValueAttrs, DynamicTensorAccessor> const
        &input_tensors,
    Allocator &allocator,
    LocalTaskRegistry &task_registry,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &device_handle,
    DeviceType kernel_device_type,
    FFIterationConfig const &iteration_config,
    size_t device_idx) {
  DynamicOpenDataflowGraph const &expanded_dg =
      instance.expanded_dataflow_graph;
  ASSERT(no_tensors_are_allocated(expanded_dg));
  DynamicOpenDataflowGraph allocated_dg =
      perform_tensor_allocation(expanded_dg, input_tensors, allocator);
  ASSERT(all_tensors_are_allocated(expanded_dg));

  // Initialize all operators and save the per-device op state
  ASSERT(no_nodes_are_initialized(allocated_dg));
  DynamicOpenDataflowGraph initialized_dg = transform_dynamic_invocation_set(
      allocated_dg, [&](DynamicNodeInvocation const &invocation) {
        return initialize_node(invocation,
                               allocator,
                               task_registry,
                               profiling_settings,
                               device_handle,
                               kernel_device_type,
                               iteration_config,
                               device_idx);
      });

  ASSERT(all_nodes_are_initialized(initialized_dg));

  return InitializedComputationGraphInstance{
      initialized_dg, allocator, task_registry};
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_forward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    perform_backward_pass_for_computation_graph_instance(
        InitializedComputationGraphInstance const &instance) {

  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
