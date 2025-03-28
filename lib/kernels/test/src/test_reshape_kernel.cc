#include "doctest/doctest.h"
#include "kernels/reshape_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Reshape Forward and Backward") {
    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    TensorShape input_shape = make_float_tensor_shape_from_legion_dims({100_n});
    TensorShape output_shape = input_shape;

    ReshapePerDeviceState state =
        Kernels::Reshape::init_kernel(DataType::FLOAT);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(input_shape, allocator, 1.0f));
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Reshape::forward_kernel(
          managed_stream.raw_stream(), state, input_accessor, output_accessor);

      std::vector<float> check_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      std::vector<float> expected_output_data(
          input_accessor.shape.num_elements().unwrap_nonnegative(), 1.0f);
      CHECK(check_output_data == expected_output_data);
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          read_only_accessor_from_write_accessor(
              create_filled_accessor_w(output_shape, allocator, 1.0f));
      GenericTensorAccessorW input_grad_accessor =
          create_filled_accessor_w(input_shape, allocator, 2.0f);

      Kernels::Reshape::backward_kernel(managed_stream.raw_stream(),
                                        state,
                                        input_grad_accessor,
                                        output_grad_accessor);

      std::vector<float> host_grad_input_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(input_grad_accessor));

      std::vector<float> expected_grad_input_data(
          input_grad_accessor.shape.num_elements().unwrap_nonnegative(), 3.0f);
      CHECK(host_grad_input_data == expected_grad_input_data);
    }
  }
}
