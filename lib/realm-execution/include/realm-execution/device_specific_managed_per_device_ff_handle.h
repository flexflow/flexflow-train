#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DEVICE_SPECIFIC_MANAGED_PER_DEVICE_FF_HANDLE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DEVICE_SPECIFIC_MANAGED_PER_DEVICE_FF_HANDLE_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "pcg/device_id_t.dtg.h"
#include "realm-execution/tasks/serializer/serializable_device_specific_ptr.dtg.h"
#include <nlohmann/json.hpp>
#include <optional>

namespace FlexFlow {

struct DeviceSpecificManagedPerDeviceFFHandle {
public:
  DeviceSpecificManagedPerDeviceFFHandle() = delete;
  explicit DeviceSpecificManagedPerDeviceFFHandle(
      device_id_t owner, std::optional<ManagedPerDeviceFFHandle *> handle);

  std::optional<ManagedPerDeviceFFHandle *> get(device_id_t device_idx) const;

  SerializableDeviceSpecificPtr serialize() const;
  static DeviceSpecificManagedPerDeviceFFHandle
      deserialize(SerializableDeviceSpecificPtr const &j);

private:
  device_id_t owner;
  std::optional<ManagedPerDeviceFFHandle *> handle;
};

DeviceSpecificManagedPerDeviceFFHandle make_device_specific_managed_handle(
    device_id_t const &, std::optional<ManagedPerDeviceFFHandle *> const &);

device_handle_t device_handle_t_from_device_specific_managed_handle(
    DeviceSpecificManagedPerDeviceFFHandle const &, device_id_t);

} // namespace FlexFlow

#endif
