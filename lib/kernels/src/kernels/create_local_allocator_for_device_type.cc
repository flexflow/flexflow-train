#include "kernels/create_local_allocator_for_device_type.h"
#include "kernels/local_cpu_allocator.h"
#include "kernels/local_cuda_allocator.h"

namespace FlexFlow {

Allocator create_local_allocator_for_device_type(DeviceType device_type) {
  if (device_type == DeviceType::GPU) {
    return create_local_cuda_memory_allocator();
  } else {
    ASSERT(device_type == DeviceType::CPU);
    return create_local_cpu_memory_allocator();
  }
}

} // namespace FlexFlow
