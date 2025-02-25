#include "doctest/doctest.h"
#include "kernels/local_cpu_allocator.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test LocalCPUAllocator") {
    Allocator cpu_allocator = create_local_cpu_memory_allocator();

    SUBCASE("Test allocate and deallocate") {
      void *ptr = cpu_allocator.allocate(100);
      CHECK(ptr != nullptr);
      cpu_allocator.deallocate(ptr);
    }

    SUBCASE("Test get_allocation_device_type") {
      CHECK(cpu_allocator.get_allocation_device_type() == DeviceType::CPU);
    }
  }
}
