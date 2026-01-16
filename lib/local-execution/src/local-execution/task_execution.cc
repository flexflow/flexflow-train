#include "local-execution/task_execution.h"
#include "local-execution/local_task_argument_accessor.h"

namespace FlexFlow {

TaskArgumentAccessor make_task_argument_accessor_for_invocation(
    DynamicNodeInvocation const &invocation,
    Allocator &allocator,
    ProfilingSettings const &profiling_settings,
    DeviceType kernel_device_type,
    PCGOperatorAttrs op_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    FFIterationConfig iteration_config,
    std::optional<OptimizerAttrs> const &optimizer_attrs) {
  std::unordered_map <

      return TaskArgumentAccessor::create<LocalTaskArgumentAccessor>(
          /*allocator=*/allocator,
          /*tensor_slots_backing=*/
      );
}

} // namespace FlexFlow
