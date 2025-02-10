#include "doctest/doctest.h"
#include "kernels/pool_2d_kernels.h"
#include "op-attrs/datatype_value.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Pool2D Forward and Backward Kernel") {
    nonnegative_int input_w = 10_n;
    nonnegative_int input_h = 10_n;
    nonnegative_int input_c = 3_n;
    nonnegative_int input_n = 1_n;
    nonnegative_int output_w = 5_n;
    nonnegative_int output_h = 5_n;
    nonnegative_int output_c = 3_n;
    nonnegative_int output_n = 1_n;
    nonnegative_int pad_h = 0_n;
    nonnegative_int pad_w = 0_n;
    nonnegative_int kernel_h = 2_n;
    nonnegative_int kernel_w = 2_n;
    nonnegative_int stride_h = 2_n;
    nonnegative_int stride_w = 2_n;

    PoolOp pool_type = PoolOp::MAX;

    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    Pool2DPerDeviceState state =
        Kernels::Pool2D::init_kernel(/*handle=*/managed_handle.raw_handle(),
                                     /*activation=*/std::nullopt,
                                     /*input_w=*/input_w.unwrap_nonnegative(),
                                     /*input_h=*/input_h.unwrap_nonnegative(),
                                     /*input_c=*/input_c.unwrap_nonnegative(),
                                     /*input_n=*/input_n.unwrap_nonnegative(),
                                     /*output_w=*/output_w.unwrap_nonnegative(),
                                     /*output_h=*/output_h.unwrap_nonnegative(),
                                     /*output_c=*/output_c.unwrap_nonnegative(),
                                     /*output_n=*/output_n.unwrap_nonnegative(),
                                     /*pad_h=*/pad_h.unwrap_nonnegative(),
                                     /*pad_w=*/pad_w.unwrap_nonnegative(),
                                     /*kernel_h=*/kernel_h.unwrap_nonnegative(),
                                     /*kernel_w=*/kernel_w.unwrap_nonnegative(),
                                     /*stride_h=*/stride_h.unwrap_nonnegative(),
                                     /*stride_w=*/stride_w.unwrap_nonnegative(),
                                     /*pool_type=*/pool_type);

    TensorShape input_shape = make_tensor_shape_from_legion_dims(
        {input_w, input_h, input_c, input_n}, DataType::FLOAT);
    TensorShape output_shape = make_tensor_shape_from_legion_dims(
        {output_w, output_h, output_c, output_n}, DataType::FLOAT);

    GenericTensorAccessorW input_accessor =
        create_random_filled_accessor_w(input_shape, allocator);
    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);

    SUBCASE("forward_kernel") {
      Kernels::Pool2D::forward_kernel(managed_stream.raw_stream(),
                                      state,
                                      input_accessor.ptr,
                                      output_accessor.ptr);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_accessor = create_filled_accessor_w(
          output_shape, allocator, make_float_data_type_value(1));
      GenericTensorAccessorW input_grad_accessor =
          allocator.allocate_tensor(input_shape);

      Kernels::Pool2D::backward_kernel(managed_stream.raw_stream(),
                                       state,
                                       output_accessor.ptr,
                                       output_grad_accessor.ptr,
                                       input_accessor.ptr,
                                       input_grad_accessor.ptr);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
