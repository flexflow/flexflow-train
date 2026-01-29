#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_TASK_EXECUTION_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_TASK_EXECUTION_H

#include "kernels/profiling_settings.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "task-spec/dynamic_graph/dynamic_node_invocation.dtg.h"
#include "task-spec/per_device_op_state.dtg.h"
#include "task-spec/task_argument_accessor/task_argument_accessor.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

TaskArgumentAccessor make_task_argument_accessor_for_invocation(
    DynamicNodeInvocation const &,
    Allocator &,
    ProfilingSettings const &,
    device_handle_t const &,
    std::optional<LossAttrs> const &,
    std::optional<PerDeviceOpState> const &,
    FFIterationConfig,
    std::optional<OptimizerAttrs> const &,
    device_id_t);

std::optional<milliseconds_t>
    execute_dynamic_node_invocation(DynamicNodeInvocation const &,
                                    Allocator &,
                                    ProfilingSettings const &,
                                    device_handle_t const &,
                                    std::optional<LossAttrs> const &,
                                    std::optional<PerDeviceOpState> const &,
                                    FFIterationConfig,
                                    std::optional<OptimizerAttrs> const &,
                                    device_id_t);

} // namespace FlexFlow

#endif
