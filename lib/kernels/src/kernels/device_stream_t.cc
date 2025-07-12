#include "kernels/device_stream_t.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

device_stream_t get_gpu_device_stream() {
  ffStream_t stream;
  checkCUDA(get_legion_stream(&stream));
  return device_stream_t{stream};
}

device_stream_t get_cpu_device_stream() {
  return device_stream_t{std::monostate{}};
}

device_stream_t get_stream_for_device_type(DeviceType device_type) {
  if (device_type == DeviceType::GPU) {
    return get_gpu_device_stream();
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return get_cpu_device_stream();
  }
}

} // namespace FlexFlow
