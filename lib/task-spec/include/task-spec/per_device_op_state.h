#ifndef _FLEXFLOW_LOCAL_EXECUTION_PER_DEVICE_STATE_H
#define _FLEXFLOW_LOCAL_EXECUTION_PER_DEVICE_STATE_H

#include "task-spec/device_specific_device_states.dtg.h"
#include "task-spec/per_device_op_state.dtg.h"

namespace FlexFlow {

PerDeviceOpState
    get_device_state_from_device_specific(DeviceSpecificDeviceStates const &,
                                          size_t device_idx);

}

#endif
