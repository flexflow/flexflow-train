#include "doctest/doctest.h"
#include "kernels/flat_kernels.h"
#include "op-attrs/make_datatype_value.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Flat Kernel") {
    Allocator allocator = create_local_cuda_memory_allocator();

    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};
    ManagedFFStream managed_stream{};

    TensorShape input_shape =
        make_tensor_shape_from_legion_dims({100}, DataType::FLOAT);
    TensorShape output_shape = input_shape;

    GenericTensorAccessorR input_accessor =
        read_only_accessor_from_write_accessor(create_filled_accessor_w(
            input_shape, allocator, make_float_data_type_value(2)));

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Flat::forward_kernel(managed_stream.raw_stream(),
                                    input_accessor,
                                    output_accessor.get_float_ptr());

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor = create_filled_accessor_r(
          output_shape, allocator, make_float_data_type_value(0));
      GenericTensorAccessorW input_grad_accessor = create_filled_accessor_w(
          input_shape, allocator, make_float_data_type_value(1));

      Kernels::Flat::backward_kernel(managed_stream.raw_stream(),
                                     input_accessor,
                                     output_grad_accessor.get_float_ptr(),
                                     input_grad_accessor.get_float_ptr());

      CHECK(contains_non_zero(input_grad_accessor));
    }
  }
}
