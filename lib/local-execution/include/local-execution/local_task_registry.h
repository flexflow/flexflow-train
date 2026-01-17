#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_REGISTRY_H

#include "pcg/layer_attrs.dtg.h"
#include "task-spec/device_specific_per_device_op_state.dtg.h"
#include "task-spec/task_impl_function.dtg.h"
#include "utils/units/milliseconds_t.h"
#include <optional>

namespace FlexFlow {

std::optional<TaskImplFunction>
    get_init_task_impl_for_op_attrs(ComputationGraphOpAttrs const &);
TaskImplFunction
    get_fwd_task_impl_for_op_attrs(ComputationGraphOpAttrs const &);
TaskImplFunction
    get_bwd_task_impl_for_op_attrs(ComputationGraphOpAttrs const &);

std::optional<DeviceSpecificPerDeviceOpState>
    call_init_task_impl(ComputationGraphOpAttrs const &,
                        TaskArgumentAccessor const &arg_accessor);

std::optional<milliseconds_t>
    call_fwb_task_impl(ComputationGraphOpAttrs const &,
                       TaskArgumentAccessor const &arg_accessor);

void call_generic_task_impl(ComputationGraphOpAttrs const &,
                            TaskArgumentAccessor const &arg_accessor);

} // namespace FlexFlow

#endif
