#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_TO_TASK_INVOCATION_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_TO_TASK_INVOCATION_H

#include "local-execution/device_specific_device_states.dtg.h"
#include "local-execution/op_task_invocation.h"
#include "local-execution/runtime_arg_config.h"
#include "local-execution/task_invocation.dtg.h"
#include "pcg/computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"

namespace FlexFlow {

TaskInvocation
    lower_to_task_invocation(OpTaskInvocation const &,
                             layer_guid_t const &,
                             ComputationGraph const &,
                             std::optional<DeviceSpecificDeviceStates> const &);

ConcreteArgSpec lower_to_concrete_arg_spec(RuntimeArgRefSpec const &,
                                           RuntimeArgConfig const &);

ConcreteArgSpec lower_to_concrete_arg_spec(
    OpArgRefSpec const &,
    ComputationGraph const &,
    layer_guid_t const &,
    std::optional<DeviceSpecificDeviceStates> const &);

} // namespace FlexFlow

#endif
