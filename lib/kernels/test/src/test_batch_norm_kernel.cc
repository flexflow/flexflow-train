#include "internal/test_utils.h"
#include "kernels/batch_norm_kernels.h"
#include "op-attrs/datatype_value.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test BatchNorm Kernel") {
    positive_int output_n = 1_p;
    positive_int output_c = 10_p;
    positive_int output_h = 10_p;
    positive_int output_w = 10_p;

    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    Allocator allocator = create_local_cuda_memory_allocator();

    BatchNormPerDeviceState state = Kernels::BatchNorm::init_kernel(
        /*handle=*/managed_handle.raw_handle(),
        /*allocator=*/allocator,
        /*runningMean=*/nullptr,
        /*output_n=*/output_n.int_from_positive_int(),
        /*output_c=*/output_c.int_from_positive_int(),
        /*output_h=*/output_h.int_from_positive_int(),
        /*output_w=*/output_w.int_from_positive_int(),
        /*relu=*/true);

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{output_n, output_c, output_h, output_w}},
        DataType::FLOAT,
    };
    TensorShape output_shape = TensorShape{
        TensorDims{FFOrdered{output_n, output_c, output_h, output_w}},
        DataType::FLOAT,
    };
    TensorShape scale_shape = TensorShape{
        TensorDims{FFOrdered{output_n, output_c, output_h, output_w}},
        DataType::FLOAT,
    };
    TensorShape bias_shape = TensorShape{
        TensorDims{FFOrdered{output_n, output_c, output_h, output_w}},
        DataType::FLOAT,
    };

    GenericTensorAccessorW input_accessor =
        create_random_filled_accessor_w(input_shape, allocator);
    GenericTensorAccessorW output_accessor =
        create_random_filled_accessor_w(output_shape, allocator);
    GenericTensorAccessorW scale_accessor = create_filled_accessor_w(
        scale_shape, allocator, make_float_data_type_value(1));

    SUBCASE("forward_kernel") {
      GenericTensorAccessorW bias_accessor = create_filled_accessor_w(
          bias_shape, allocator, make_float_data_type_value(0));

      Kernels::BatchNorm::forward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*per_device_state=*/state,
          /*input_ptr=*/input_accessor.get_float_ptr(),
          /*output_ptr=*/output_accessor.get_float_ptr(),
          /*scale_ptr=*/scale_accessor.get_float_ptr(),
          /*bias_ptr=*/bias_accessor.get_float_ptr());

      CHECK(contains_non_zero(output_accessor));
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
          /*output_ptr=*/output_accessor.get_float_ptr(),
          /*output_grad_ptr=*/output_grad_accessor.get_float_ptr(),
          /*input_ptr=*/input_accessor.get_float_ptr(),
          /*input_grad_ptr=*/input_grad_accessor.get_float_ptr(),
          /*scale_ptr=*/scale_accessor.get_float_ptr(),
          /*scale_grad_ptr=*/scale_grad_accessor.get_float_ptr(),
          /*bias_grad_ptr=*/bias_grad_accessor.get_float_ptr(),
          /*numElements=*/
          input_accessor.shape.num_elements().int_from_positive_int());

      CHECK(contains_non_zero(input_grad_accessor));
      CHECK(contains_non_zero(scale_grad_accessor));
      CHECK(contains_non_zero(bias_grad_accessor));
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
