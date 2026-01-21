#include "local-execution/task_execution.h"
#include "local-execution/local_task_argument_accessor.h"
#include "utils/exception.h"

namespace FlexFlow {

TaskArgumentAccessor make_task_argument_accessor_for_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    device_handle_t const &ff_handle,
    DeviceType kernel_device_type,
    PCGOperatorAttrs op_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    FFIterationConfig iteration_config,
    std::optional<OptimizerAttrs> const &optimizer_attrs,
    size_t device_idx) {
  std::unordered_map<TaskTensorParameter, DynamicTensorAccessor>
      tensor_slots_backing;

  return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
      /*allocator=*/allocator,
      /*tensor_slots_backing=*/tensor_slots_backing,
      /*profiling_settings=*/profiling_settings,
      /*ff_handle=*/ff_handle,
      /*kernel_device_type=*/kernel_device_type,
      /*op_attrs=*/op_attrs,
      /*loss_attrs=*/loss_attrs,
      /*per_device_op_state=*/per_device_op_state,
      /*iteration_config=*/iteration_config,
      /*optimizer_attrs=*/optimizer_attrs,
      /*device_idx=*/device_idx);
}

std::optional<milliseconds_t> execute_dynamic_node_invocation(
    DynamicNodeInvocation const &invocation,
    ProfilingSettings const &profiling_settings,
    DeviceType kernel_device_type,
    PCGOperatorAttrs op_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    FFIterationConfig iteration_config,
    std::optional<OptimizerAttrs> const &optimizer_attrs) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
