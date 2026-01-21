#include "local-execution/task_execution.h"
#include "local-execution/local_task_argument_accessor.h"
#include "local-execution/local_task_registry.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "utils/exception.h"
#include "utils/optional.h"
#include <optional>

namespace FlexFlow {

TaskArgumentAccessor make_task_argument_accessor_for_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    FFIterationConfig iteration_config,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    device_id_t device_idx) {
  PCGOperatorAttrs op_attrs = assert_unwrap(invocation.node_attrs.op_attrs);

  std::unordered_map<TaskTensorParameter, DynamicTensorAccessor>
      tensor_slots_backing;
  NOT_IMPLEMENTED(); // FIXME (Elliott): fill the map

  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      /*allocator=*/allocator,
      /*tensor_slots_backing=*/tensor_slots_backing,
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*op_attrs=*/op_attrs,
      /*loss_attrs=*/loss_attrs,
      /*per_device_op_state=*/per_device_op_state,
      /*iteration_config=*/iteration_config,
      /*optimizer_attrs=*/optimizer_attrs,
      /*device_idx=*/device_idx);
}

std::optional<milliseconds_t> execute_dynamic_node_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    FFIterationConfig iteration_config,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    device_id_t device_idx) {
  TaskArgumentAccessor arg_accessor =
      make_task_argument_accessor_for_invocation(
          /*invocation=*/invocation,
          /*allocator=*/allocator,
          /*profiling_settings=*/profiling_settings,
          /*ff_handle=*/ff_handle,
          /*loss_attrs=*/loss_attrs,
          /*per_device_op_state=*/per_device_op_state,
          /*iteration_config=*/iteration_config,
          /*optimizer_attrs=*/optimizer_attrs,
          /*device_idx=*/device_idx);

  DynamicTaskType task_type = assert_unwrap(invocation.node_attrs.task_type);
  ComputationGraphOpAttrs op_attrs =
      assert_unwrap(compgraph_op_attrs_from_pcg_op_attrs(
          assert_unwrap(invocation.node_attrs.op_attrs)));
  std::optional<TaskImplFunction> task_impl;
  switch (task_type) {
    case DynamicTaskType::FWD:
      task_impl = get_fwd_task_impl_for_op_attrs(op_attrs);
      break;
    case DynamicTaskType::BWD:
      task_impl = get_bwd_task_impl_for_op_attrs(op_attrs);
      break;
    case DynamicTaskType::UPD:
      NOT_IMPLEMENTED();
      break;
    default:
      PANIC("Unhandled DynamicTaskType", fmt::to_string(task_type));
  }
  if (!task_impl) {
    return std::nullopt;
  }
  NOT_IMPLEMENTED(); // FIXME (Elliott): call the task
  return std::nullopt;
}

} // namespace FlexFlow
