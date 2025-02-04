#include "doctest/doctest.h"
#include "kernels/concat_kernels.h"
#include "test_utils.h"
#include "utils/containers/repeat.h"

using namespace ::FlexFlow;
TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test concat kernel forward and backward") {
    nonnegative_int num_inputs = 3_n;
    nonnegative_int size_per_input = 100_n;
    ff_dim_t concat_axis = ff_dim_t{0_n};

    ManagedPerDeviceFFHandle managed_handle{};
    ManagedFFStream managed_stream{};

    TensorShape input_shape =
        make_float_tensor_shape_from_legion_dims({size_per_input});
    TensorShape output_shape =
        make_float_tensor_shape_from_legion_dims({size_per_input, num_inputs});

    Allocator allocator = create_local_cuda_memory_allocator();

    SUBCASE("forward_kernel") {
      std::vector<GenericTensorAccessorR> input_accessors =
          repeat(num_inputs, [&]() {
            return read_only_accessor_from_write_accessor(
                create_random_filled_accessor_w(input_shape, allocator));
          });
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Concat::forward_kernel(managed_stream.raw_stream(),
                                      output_accessor,
                                      input_accessors,
                                      concat_axis);

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));

      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorR output_grad_accessor =
          read_only_accessor_from_write_accessor(
              create_random_filled_accessor_w(output_shape, allocator));
      std::vector<GenericTensorAccessorW> input_grad_accessors = repeat(
          num_inputs, [&]() { return allocator.allocate_tensor(input_shape); });
      Kernels::Concat::backward_kernel(managed_stream.raw_stream(),
                                       output_grad_accessor,
                                       input_grad_accessors,
                                       concat_axis);
    }
  }
}
