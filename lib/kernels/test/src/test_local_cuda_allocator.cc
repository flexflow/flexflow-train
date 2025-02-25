#include "kernels/local_cuda_allocator.h"
#include "doctest/doctest.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test LocalCUDAAllocator") {
    Allocator gpu_allocator = create_local_cuda_memory_allocator();

    SUBCASE("Test allocate and deallocate") {
      void *ptr = gpu_allocator.allocate(100);
      CHECK(ptr != nullptr);
      gpu_allocator.deallocate(ptr);
    }

    SUBCASE("Test get_allocation_device_type") {
      CHECK(gpu_allocator.get_allocation_device_type() == DeviceType::GPU);
    }
  }
}
