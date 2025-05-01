#include "internal/test_utils.h"
#include "kernels/reduction_kernels.h"
#include "op-attrs/datatype_value.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Reduction Forward and Backward Kernel") {
    std::size_t num_replicas = 5;

    TensorShape input_shape = make_tensor_shape(
        FFOrdered{10_n, 10_n, 10_n, 10_n, 10_n}, DataType::FLOAT);

    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    SUBCASE("forward_kernel") {
      TensorShape output_shape =
          make_tensor_shape(FFOrdered{10_n}, DataType::FLOAT);

      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Reduction::forward_kernel(managed_stream.raw_stream(),
                                         input_accessor,
                                         output_accessor,
                                         num_replicas);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      TensorShape output_shape = input_shape;

      GenericTensorAccessorR output_grad_accessor = create_filled_accessor_r(
          output_shape, allocator, make_float_data_type_value(1));
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Reduction::backward_kernel(managed_stream.raw_stream(),
                                          output_grad_accessor,
                                          input_grad_accessor);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
