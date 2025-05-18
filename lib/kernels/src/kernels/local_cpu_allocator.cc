#include "kernels/local_cpu_allocator.h"
#include "kernels/device.h"
#include "utils/containers/contains_key.h"
#include <libassert/assert.hpp>
#include <stdlib.h>

namespace FlexFlow {
void *LocalCPUAllocator::allocate(size_t requested_memory_size) {
  void *ptr = malloc(requested_memory_size);
  ASSERT(ptr != nullptr);
  this->ptrs.insert({ptr, std::unique_ptr<void, decltype(&free)>(ptr, free)});
  return ptr;
}

void LocalCPUAllocator::deallocate(void *ptr) {
  ASSERT(contains_key(this->ptrs, ptr),
         "Deallocating a pointer that was not allocated by this Allocator");

  free(ptr);
  this->ptrs.erase(ptr);
}

DeviceType LocalCPUAllocator::get_allocation_device_type() const {
  return DeviceType::CPU;
}

Allocator create_local_cpu_memory_allocator() {
  return Allocator::create<LocalCPUAllocator>();
}

} // namespace FlexFlow
