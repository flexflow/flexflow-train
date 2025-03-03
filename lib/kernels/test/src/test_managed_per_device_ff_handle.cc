#include "doctest/doctest.h"
#include "kernels/managed_per_device_ff_handle.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test ManagedPerDeviceFFHandle") {
    ManagedPerDeviceFFHandle base_handle{/*workSpaceSize=*/1024 * 1024,
                                         /*allowTensorOpMathConversion=*/true};
    PerDeviceFFHandle const *base_handle_ptr = &base_handle.raw_handle();

    SUBCASE("constructor") {
      CHECK(base_handle.raw_handle().workSpaceSize == 1024 * 1024);
      CHECK(base_handle.raw_handle().allowTensorOpMathConversion == true);
    }

    SUBCASE("move constructor") {
      ManagedPerDeviceFFHandle new_handle(std::move(base_handle));
      CHECK(&new_handle.raw_handle() == base_handle_ptr);
    }

    SUBCASE("move assignment operator") {
      SUBCASE("move assign to other") {
        ManagedPerDeviceFFHandle new_handle{
            /*workSpaceSize=*/1024 * 1024,
            /*allowTensorOpMathConversion=*/true};
        new_handle = std::move(base_handle);
        CHECK(&new_handle.raw_handle() == base_handle_ptr);
      }

      SUBCASE("move assign to self") {
        base_handle = std::move(base_handle);
        CHECK(&base_handle.raw_handle() == base_handle_ptr);
      }
    }
  }
}
