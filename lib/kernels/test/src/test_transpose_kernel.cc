#include "internal/test_utils.h"
#include "kernels/transpose_kernels.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Transpose Kernel Operations") {
    TransposeAttrs attrs = TransposeAttrs{
        FFOrdered{
            ff_dim_t{1_n},
            ff_dim_t{0_n},
        },
    };

    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{10_p, 10_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Transpose::forward_kernel(
          managed_stream.raw_stream(), attrs, input_accessor, output_accessor);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          create_random_filled_accessor_r(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          create_random_filled_accessor_w(input_shape, allocator);

      Kernels::Transpose::backward_kernel(managed_stream.raw_stream(),
                                          attrs,
                                          output_grad_accessor,
                                          input_grad_accessor);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
