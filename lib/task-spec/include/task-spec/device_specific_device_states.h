#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DEVICE_SPECIFIC_DEVICE_STATES_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DEVICE_SPECIFIC_DEVICE_STATES_H

#include "task-spec/device_specific_device_states.dtg.h"

namespace FlexFlow {

template <typename T>
std::optional<DeviceSpecificDeviceStates> 
  make_device_specific_state(std::optional<T> const &per_device_state) {
  if (!per_device_state.has_value()) {
    return std::nullopt;
  }

  return DeviceSpecificDeviceStates{
    DeviceSpecific<T>::create(per_device_state.value()),
  };
}

} // namespace FlexFlow

#endif
