#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_TO_TASK_INVOCATION_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_TO_TASK_INVOCATION_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/layer_guid_t.dtg.h"
#include "task-spec/device_specific_device_states.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/runtime_arg_config.h"
#include "task-spec/task_invocation.dtg.h"

namespace FlexFlow {

TaskInvocation lower_to_task_invocation(
    OpTaskInvocation const &,
    layer_guid_t const &,
    ComputationGraph const &,
    std::unordered_map<tensor_guid_t, gradient_tensor_t> const &,
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
