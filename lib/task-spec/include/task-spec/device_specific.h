#ifndef _FLEXFLOW_LOCAL_EXECUTION_DEVICE_SPECIFIC_H
#define _FLEXFLOW_LOCAL_EXECUTION_DEVICE_SPECIFIC_H

#include "pcg/device_id_t.dtg.h"
#include "task-spec/serialization.h"

namespace FlexFlow {

template <typename T>
struct DeviceSpecific {
  DeviceSpecific() = delete;

  template <typename... Args>
  static DeviceSpecific<T> create(device_id_t device_idx, Args &&...args) {
    return DeviceSpecific<T>(std::make_shared<T>(std::forward<Args>(args)...),
                             device_idx);
  }

  bool operator==(DeviceSpecific const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(DeviceSpecific const &other) const {
    return this->tie() != other.tie();
  }

  T const *get(device_id_t curr_device_idx) const {
    ASSERT(curr_device_idx == this->device_idx);
    return (T const *)this->ptr.get();
  }
private:
  DeviceSpecific(std::shared_ptr<T> ptr, device_id_t device_idx)
      : ptr(ptr), device_idx(device_idx) {}

private:
  std::shared_ptr<T> ptr;
  device_id_t device_idx;

private:
  std::tuple<decltype(ptr) const &, decltype(device_idx) const &> tie() const {
    return std::tie(this->ptr, this->device_idx);
  }
};

// manually force serialization to make DeviceSpecific trivially
// serializable
// template <typename T>
// struct is_trivially_serializable<DeviceSpecific<T>> : std::true_type {};

} // namespace FlexFlow

#endif
