#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "kernels/device_handle_t.h"
#include "utils/containers/transform.h"
#include "utils/json/optional.h"
#include <cstdint>

namespace FlexFlow {

DeviceSpecificManagedPerDeviceFFHandle::DeviceSpecificManagedPerDeviceFFHandle(
    device_id_t owner, std::optional<ManagedPerDeviceFFHandle *> handle)
    : owner(owner), handle(handle) {}

std::optional<ManagedPerDeviceFFHandle *>
    DeviceSpecificManagedPerDeviceFFHandle::get(device_id_t device_idx) const {
  ASSERT(this->owner == device_idx);
  return this->handle;
}

SerializableDeviceSpecificPtr
    DeviceSpecificManagedPerDeviceFFHandle::serialize() const {
  return SerializableDeviceSpecificPtr{
      /*device_idx=*/owner,
      /*ptr=*/
      transform(handle,
                [](ManagedPerDeviceFFHandle *ptr) {
                  return reinterpret_cast<uintptr_t>(ptr);
                }),
  };
}

DeviceSpecificManagedPerDeviceFFHandle
    DeviceSpecificManagedPerDeviceFFHandle::deserialize(
        SerializableDeviceSpecificPtr const &handle) {
  return DeviceSpecificManagedPerDeviceFFHandle{
      /*owner=*/handle.device_idx,
      /*handle=*/
      transform(handle.ptr,
                [](uintptr_t ptrval) {
                  return reinterpret_cast<ManagedPerDeviceFFHandle *>(ptrval);
                }),
  };
}

DeviceSpecificManagedPerDeviceFFHandle make_device_specific_managed_handle(
    device_id_t const &device_id,
    std::optional<ManagedPerDeviceFFHandle *> const &managed_handle) {
  return DeviceSpecificManagedPerDeviceFFHandle{device_id, managed_handle};
}

device_handle_t device_handle_t_from_device_specific_managed_handle(
    DeviceSpecificManagedPerDeviceFFHandle const &device_specific,
    device_id_t device_idx) {
  return device_handle_t_from_managed_handle_ptr(
      *device_specific.get(device_idx));
}

} // namespace FlexFlow
