#include "doctest/doctest.h"
#include "kernels/batch_matmul_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test BatchMatmul Kernel") {
    nonnegative_int m = 10_n;
    nonnegative_int n = 10_n;
    nonnegative_int k = 10_n;
    nonnegative_int batch = 5_n;
    int a_seq_length_dim = -1;
    int b_seq_length_dim = -1;
    int seq_length = -1;

    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape_a =
        make_tensor_shape_from_legion_dims({m, k, batch}, DataType::FLOAT);
    TensorShape input_shape_b =
        make_tensor_shape_from_legion_dims({k, n, batch}, DataType::FLOAT);
    TensorShape output_shape =
        make_tensor_shape_from_legion_dims({m, n, batch}, DataType::FLOAT);

    GenericTensorAccessorW a_accessor =
        create_random_filled_accessor_w(input_shape_a, allocator);
    GenericTensorAccessorW b_accessor =
        create_random_filled_accessor_w(input_shape_b, allocator);
    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);

    SUBCASE("forward_kernel") {
      Kernels::BatchMatmul::forward_kernel(managed_stream.raw_stream(),
                                           managed_handle.raw_handle(),
                                           output_accessor.get_float_ptr(),
                                           a_accessor.get_float_ptr(),
                                           b_accessor.get_float_ptr(),
                                           m.unwrap_nonnegative(),
                                           n.unwrap_nonnegative(),
                                           k.unwrap_nonnegative(),
                                           batch.unwrap_nonnegative(),
                                           a_seq_length_dim,
                                           b_seq_length_dim,
                                           seq_length);
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW o_grad_accessor =
          create_random_filled_accessor_w(output_shape, allocator);
      GenericTensorAccessorW a_grad_accessor =
          allocator.allocate_tensor(input_shape_a);
      GenericTensorAccessorW b_grad_accessor =
          allocator.allocate_tensor(input_shape_b);

      Kernels::BatchMatmul::backward_kernel(managed_stream.raw_stream(),
                                            managed_handle.raw_handle(),
                                            output_accessor.get_float_ptr(),
                                            o_grad_accessor.get_float_ptr(),
                                            a_accessor.get_float_ptr(),
                                            a_grad_accessor.get_float_ptr(),
                                            b_accessor.get_float_ptr(),
                                            b_grad_accessor.get_float_ptr(),
                                            m.unwrap_nonnegative(),
                                            n.unwrap_nonnegative(),
                                            k.unwrap_nonnegative(),
                                            batch.unwrap_nonnegative());
    }
  }
}
