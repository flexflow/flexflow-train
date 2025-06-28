#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PER_DEVICE_OP_STATE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PER_DEVICE_OP_STATE_H

#include "task-spec/concrete_arg_spec.h"
#include "task-spec/device_specific_device_states.dtg.h"
#include "task-spec/per_device_op_state.dtg.h"
#include "utils/type_index.h"

namespace FlexFlow {

PerDeviceOpState
    get_device_state_from_device_specific(DeviceSpecificDeviceStates const &,
                                          size_t device_idx);

}

#endif
