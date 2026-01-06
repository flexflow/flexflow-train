#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_TASK_EXECUTION_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_TASK_EXECUTION_H

#include "kernels/profiling_settings.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "task-spec/per_device_op_state.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"

namespace FlexFlow {

TaskArgumentAccessor
  make_task_argument_accessor_for_invocation(
    DynamicNodeInvocation const &invocation,
    ProfilingSettings const &profiling_settings,
    DeviceType kernel_device_type,
    PCGOperatorAttrs op_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    FFIterationConfig iteration_config,
    std::optional<OptimizerAttrs> const &optimizer_attrs);

void
  execute_dynamic_node_invocation(
    DynamicNodeInvocation const &invocation,
    ProfilingSettings const &profiling_settings,
    DeviceType kernel_device_type,
    PCGOperatorAttrs op_attrs,
    std::optional<LossAttrs> const &loss_attrs,
    std::optional<PerDeviceOpState> const &per_device_op_state,
    FFIterationConfig iteration_config,
    std::optional<OptimizerAttrs> const &optimizer_attrs);

} // namespace FlexFlow

#endif
