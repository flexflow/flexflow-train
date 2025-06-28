#include "kernels/device_handle_t.h"

namespace FlexFlow {

device_handle_t gpu_make_device_handle_t(PerDeviceFFHandle const &ff_handle) {
  return device_handle_t{
      ff_handle,
  };
}

device_handle_t cpu_make_device_handle_t() {
  return device_handle_t{std::monostate{}};
}

} // namespace FlexFlow
