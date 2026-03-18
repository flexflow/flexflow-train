#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DEVICE_SPECIFIC_PTR_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_DEVICE_SPECIFIC_PTR_H

#include "pcg/device_id_t.dtg.h"
#include <optional>

namespace FlexFlow {

/**
 * \brief Holds a pointer into (potentially) remote memory which checks that
 * the memory is local on access.
 *
 * There exist per-device states (i.e., \ref PerDeviceOpState and \ref
 * FFHandle) that need to be created and managed by the central \ref
 * term-controller "controller task". Since these are opaque pointers they
 * can't be safely copied to and from the remote devices, so we instead
 * transfer the pointers back-and-forth between workers and the controller
 * task. To prevent accidentally accessing one of these pointers on the wrong
 * device (as the pointer is only valid in the memory where it was created), we
 * wrap them with \ref DeviceSpecificPtr, which holds the \ref device_idx_t
 * where the pointer was created, and any attempt to interact with the raw
 * pointer value (i.e., \ref DeviceSpecificPtr::get) checks that the current
 * device matches the original device, and throws a readable error message if
 * it does not.
 *
 * \note \ref DeviceSpecificPtr explicitly does not own the pointer that it holds, leaving
 * lifetime management up to the user of the pointer. If you want a lifetime-managed version,
 * see \ref DeviceSpecific.
 */
template <typename T>
struct DeviceSpecificPtr {
public:
  DeviceSpecificPtr() = delete;
  explicit DeviceSpecificPtr(device_id_t device_idx, std::optional<T *> ptr)
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
