#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DEVICE_SPECIFIC_PTR_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DEVICE_SPECIFIC_PTR_H

#include "pcg/device_id_t.dtg.h"
#include <optional>

namespace FlexFlow {

template <typename T>
struct DeviceSpecificPtr {
public:
  DeviceSpecificPtr() = delete;
  explicit DeviceSpecificPtr(device_id_t device_idx, std::optional<T *> handle)
      : device_idx(device_idx), ptr(ptr) {}

  std::optional<T *> get(device_id_t device_idx) const {
    ASSERT(this->device_idx == device_idx);
    return this->ptr;
  }

  device_id_t get_device_idx() const {
    return this->device_idx;
  }

  std::optional<T *> get_unsafe_raw_ptr() const {
    return this->ptr;
  }

private:
  device_id_t device_idx;
  std::optional<T *> ptr;
};

} // namespace FlexFlow

#endif
