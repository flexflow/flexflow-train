#include "internal/test_utils.h"
#include "kernels/attention_kernels.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test multi-head attention kernel") {
    nonnegative_int num_samples = 10_n;
    nonnegative_int num_heads = 4_n;
    nonnegative_int qSize = 64_n;
    nonnegative_int kSize = 64_n;
    nonnegative_int vSize = 64_n;
    nonnegative_int qProjSize = 64_n;
    nonnegative_int kProjSize = 64_n;
    nonnegative_int vProjSize = 64_n;
    nonnegative_int oProjSize = 64_n;
    nonnegative_int qoSeqLength = 20_n;
    nonnegative_int kvSeqLength = 20_n;

    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};

    Allocator allocator = create_local_cuda_memory_allocator();

    MHAPerDeviceState state = Kernels::MultiHeadAttention::init_kernel(
        managed_handle.raw_handle(),
        allocator,
        /*num_samples=*/num_samples.unwrap_nonnegative(),
        /*num_heads=*/num_heads.unwrap_nonnegative(),
        /*qSize=*/qSize.unwrap_nonnegative(),
        /*kSize=*/kSize.unwrap_nonnegative(),
        /*vSize=*/vSize.unwrap_nonnegative(),
        /*qProjSize=*/qProjSize.unwrap_nonnegative(),
        /*kProjSize=*/kProjSize.unwrap_nonnegative(),
        /*vProjSize=*/vProjSize.unwrap_nonnegative(),
        /*oProjSize=*/oProjSize.unwrap_nonnegative(),
        /*qoSeqLength=*/qoSeqLength.unwrap_nonnegative(),
        /*kvSeqLength=*/kvSeqLength.unwrap_nonnegative(),
        /*add_bias_kv=*/false);

    TensorShape query_shape = TensorShape{
        TensorDims{FFOrdered{qoSeqLength, num_samples, qSize}},
        DataType::FLOAT,
    };
    TensorShape key_shape = TensorShape{
        TensorDims{FFOrdered{kvSeqLength, num_samples, kSize}},
        DataType::FLOAT,
    };
    TensorShape value_shape = TensorShape{
        TensorDims{FFOrdered{kvSeqLength, num_samples, vSize}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{qoSeqLength, num_samples, oProjSize}},
        DataType::FLOAT,
    };
    TensorShape weight_shape = TensorShape{
        TensorDims{FFOrdered{nonnegative_int{state.weightSize}}},
        DataType::FLOAT,
    };

    GenericTensorAccessorW query_accessor =
        create_random_filled_accessor_w(query_shape, allocator);
    GenericTensorAccessorW key_accessor =
        create_random_filled_accessor_w(key_shape, allocator);
    GenericTensorAccessorW value_accessor =
        create_random_filled_accessor_w(value_shape, allocator);
    GenericTensorAccessorW weight_accessor =
        create_random_filled_accessor_w(weight_shape, allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::MultiHeadAttention::forward_kernel(
          managed_stream.raw_stream(),
          state,
          query_accessor.get_float_ptr(),
          key_accessor.get_float_ptr(),
          value_accessor.get_float_ptr(),
          weight_accessor.get_float_ptr(),
          output_accessor.get_float_ptr());

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW query_grad_accessor =
          create_random_filled_accessor_w(query_shape, allocator);
      GenericTensorAccessorW key_grad_accessor =
          create_random_filled_accessor_w(key_shape, allocator);
      GenericTensorAccessorW value_grad_accessor =
          create_random_filled_accessor_w(value_shape, allocator);
      GenericTensorAccessorW weight_grad_accessor =
          create_random_filled_accessor_w(weight_shape, allocator);
      GenericTensorAccessorW output_grad_accessor =
          create_random_filled_accessor_w(output_shape, allocator);

      Kernels::MultiHeadAttention::backward_kernel(
          managed_stream.raw_stream(),
          state,
          query_accessor.get_float_ptr(),
          query_grad_accessor.get_float_ptr(),
          key_accessor.get_float_ptr(),
          key_grad_accessor.get_float_ptr(),
          value_accessor.get_float_ptr(),
          value_grad_accessor.get_float_ptr(),
          weight_accessor.get_float_ptr(),
          weight_grad_accessor.get_float_ptr(),
          output_grad_accessor.get_float_ptr());
    }

    Kernels::MultiHeadAttention::cleanup_kernel(allocator, state);
  }
}
