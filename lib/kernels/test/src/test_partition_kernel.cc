#include "doctest/doctest.h"
#include "kernels/partition_kernels.h"
#include "op-attrs/datatype_value.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Partition Forward and Backward") {
    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    RepartitionPerDeviceState state = Kernels::Repartition::init_kernel(
        managed_handle.raw_handle(), DataType::FLOAT);

    TensorShape input_shape =
        make_tensor_shape_from_ff_ordered({10_n, 10_n}, DataType::FLOAT);
    TensorShape output_shape = input_shape;

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor = create_filled_accessor_r(
          input_shape, allocator, make_float_data_type_value(1));
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Repartition::forward_kernel(
          managed_stream.raw_stream(), state, input_accessor, output_accessor);

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor = create_filled_accessor_r(
          output_shape, allocator, make_float_data_type_value(1));
      GenericTensorAccessorW input_grad_accessor = create_filled_accessor_w(
          input_shape, allocator, make_float_data_type_value(2));

      Kernels::Repartition::backward_kernel(managed_stream.raw_stream(),
                                            state,
                                            output_grad_accessor,
                                            input_grad_accessor);

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
