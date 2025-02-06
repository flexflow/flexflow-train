#include "doctest/doctest.h"
#include "kernels/batch_norm_kernels.h"
#include "test_utils.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test BatchNorm Kernel") {
    nonnegative_int output_n = 1_n;
    nonnegative_int output_c = 10_n;
    nonnegative_int output_h = 10_n;
    nonnegative_int output_w = 10_n;

    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle{};

    Allocator allocator = create_local_cuda_memory_allocator();

    BatchNormPerDeviceState state = Kernels::BatchNorm::init_kernel(
        /*handle=*/managed_handle.raw_handle(),
        /*allocator=*/allocator,
        /*runningMean=*/nullptr,
        /*output_n=*/output_n.unwrap_nonnegative(),
        /*output_c=*/output_c.unwrap_nonnegative(),
        /*output_h=*/output_h.unwrap_nonnegative(),
        /*output_w=*/output_w.unwrap_nonnegative(),
        /*relu=*/true);

    TensorShape input_shape = make_float_tensor_shape_from_legion_dims(
        {output_n, output_c, output_h, output_w});
    TensorShape output_shape = make_float_tensor_shape_from_legion_dims(
        {output_n, output_c, output_h, output_w});
    TensorShape scale_shape = make_float_tensor_shape_from_legion_dims(
        {output_n, output_c, output_h, output_w});
    TensorShape bias_shape = make_float_tensor_shape_from_legion_dims(
        {output_n, output_c, output_h, output_w});

    GenericTensorAccessorW input_accessor =
        create_random_filled_accessor_w(input_shape, allocator);
    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);
    GenericTensorAccessorW scale_accessor =
        create_filled_accessor_w(scale_shape, allocator, 1.0f);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW bias_accessor =
          create_filled_accessor_w(bias_shape, allocator, 0.0f);

      Kernels::BatchNorm::forward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*per_device_state=*/state,
          /*input_ptr=*/input_accessor.get_float_ptr(),
          /*output_ptr=*/output_accessor.get_float_ptr(),
          /*scale_ptr=*/scale_accessor.get_float_ptr(),
          /*bias_ptr=*/bias_accessor.get_float_ptr());

      std::vector<float> host_output_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(output_accessor));
      CHECK(contains_non_zero(host_output_data));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_accessor =
          create_random_filled_accessor_w(output_shape, allocator);
      GenericTensorAccessorW input_grad_accessor =
          create_random_filled_accessor_w(input_shape, allocator);
      GenericTensorAccessorW scale_grad_accessor =
          create_random_filled_accessor_w(scale_shape, allocator);
      GenericTensorAccessorW bias_grad_accessor =
          create_random_filled_accessor_w(bias_shape, allocator);

      Kernels::BatchNorm::backward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*per_device_state=*/state,
          /*input_ptr=*/input_accessor.get_float_ptr(),
          /*output_grad_ptr=*/output_grad_accessor.get_float_ptr(),
          /*output_ptr=*/output_accessor.get_float_ptr(),
          /*input_grad_ptr=*/input_grad_accessor.get_float_ptr(),
          /*scale_ptr=*/scale_accessor.get_float_ptr(),
          /*scale_grad_ptr=*/scale_grad_accessor.get_float_ptr(),
          /*bias_grad_ptr=*/bias_grad_accessor.get_float_ptr(),
          /*numElements=*/
          input_accessor.shape.num_elements().unwrap_nonnegative());

      std::vector<float> host_input_grad_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(input_grad_accessor));
      std::vector<float> host_scale_grad_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(scale_grad_accessor));
      std::vector<float> host_bias_grad_data =
          load_data_to_host_from_device<float>(
              read_only_accessor_from_write_accessor(bias_grad_accessor));

      CHECK(contains_non_zero(host_input_grad_data));
      CHECK(contains_non_zero(host_scale_grad_data));
      CHECK(contains_non_zero(host_bias_grad_data));
    }

    Kernels::BatchNorm::cleanup_kernel(allocator,
                                       state.inputTensor,
                                       state.biasTensor,
                                       state.outputTensor,
                                       state.actiDesc,
                                       true,
                                       state.runningMean);
  }
}
