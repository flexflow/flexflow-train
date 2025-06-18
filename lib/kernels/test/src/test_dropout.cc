#include "kernels/test_utils.h"
#include "kernels/dropout_kernels.h"
#include "utils/containers/count.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("Test Dropout Kernels") {
    unsigned long long seed = 12345;
    float dropout_rate = 0.1;

    ArrayShape shape = ArrayShape{
        std::vector{10_p, 10_p},
    };

    TensorShape input_shape = TensorShape{
        TensorDims{FFOrdered{10_p, 10_p}},
        DataType::FLOAT,
    };
    TensorShape output_shape = input_shape;

    ManagedFFStream managed_stream{};
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);

    Allocator allocator = create_local_cuda_memory_allocator();

    DropoutPerDeviceState state = Kernels::Dropout::init_kernel(
        managed_handle.raw_handle(), dropout_rate, seed, shape, allocator);

    SUBCASE("forward_kernel") {
      GenericTensorAccessorR input_accessor =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorW output_accessor =
          allocator.allocate_tensor(output_shape);

      Kernels::Dropout::forward_kernel(managed_stream.raw_stream(),
                                       state,
                                       input_accessor.get_float_ptr(),
                                       output_accessor.get_float_ptr());

      CHECK(contains_non_zero(output_accessor));
    }

    SUBCASE("backward_kernel") {
      GenericTensorAccessorW output_grad_data =
          create_random_filled_accessor_w(output_shape, allocator);
      GenericTensorAccessorW input_grad_data =
          create_random_filled_accessor_w(input_shape, allocator);

      Kernels::Dropout::backward_kernel(managed_stream.raw_stream(),
                                        state,
                                        output_grad_data.get_float_ptr(),
                                        input_grad_data.get_float_ptr());
    }

    Kernels::Dropout::cleanup_kernel(allocator,
                                     state.inputTensor,
                                     state.outputTensor,
                                     state.dropoutDesc,
                                     state.dropoutStates);
  }
}
