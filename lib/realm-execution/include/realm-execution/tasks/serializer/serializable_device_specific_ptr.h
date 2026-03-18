#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_SERIALIZABLE_DEVICE_SPECIFIC_PTR_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_SERIALIZABLE_DEVICE_SPECIFIC_PTR_H

#include "realm-execution/device_specific_ptr.h"
#include "realm-execution/tasks/serializer/serializable_device_specific_ptr.dtg.h"

namespace FlexFlow {

template <typename T>
SerializableDeviceSpecificPtr device_specific_ptr_to_serializable(
    DeviceSpecificPtr<T> const &device_specific) {
  return SerializableDeviceSpecificPtr{
      /*device_idx=*/device_specific.get_device_idx(),
      /*ptr=*/
      transform(device_specific.get_unsafe_raw_ptr(),
                [](T *ptr) { return reinterpret_cast<uintptr_t>(ptr); }),
  };
}

template <typename T>
DeviceSpecificPtr<T> device_specific_ptr_from_serializable(
    SerializableDeviceSpecificPtr const &device_specific) {
  return DeviceSpecificPtr<T>{
      /*device_idx*/ device_specific.device_idx,
      /*ptr=*/transform(device_specific.ptr, [](uintptr_t ptrval) {
        return reinterpret_cast<T *>(ptrval);
      })};
}

} // namespace FlexFlow

#endif
