#include "internal/test_utils.h"
#include "kernels/reshape_kernels.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Reshape Forward and Backward") {
    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{100_n}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;

    ReshapePerDeviceState state =
        Kernels::Reshape::init_kernel(DataType::FLOAT);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Reshape::forward_kernel(
          managed_stream.raw_stream(), state, input_accessor, output_accessor);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Reshape::backward_kernel(managed_stream.raw_stream(),
                                        state,
                                        output_grad_accessor,
                                        input_grad_accessor);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
