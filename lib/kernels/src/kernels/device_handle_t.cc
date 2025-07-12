#include "kernels/device_handle_t.h"

namespace FlexFlow {

device_handle_t device_handle_t_from_managed_handle(
    std::optional<ManagedPerDeviceFFHandle> const &managed_handle) {
  if (managed_handle.has_value()) {
    return gpu_make_device_handle_t(managed_handle.value().raw_handle());
  } else {
    return cpu_make_device_handle_t();
  }
}

device_handle_t gpu_make_device_handle_t(PerDeviceFFHandle const &ff_handle) {
  return device_handle_t{
      ff_handle,
  };
}

device_handle_t cpu_make_device_handle_t() {
  return device_handle_t{std::monostate{}};
}

} // namespace FlexFlow
