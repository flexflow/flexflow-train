#include "internal/test_utils.h"
#include "kernels/gather_kernels.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test ManagedFFStream") {
    ManagedPerDeviceFFHandle managed_handle{
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true};
    ManagedFFStream managed_stream{};
    Allocator allocator = create_local_cuda_memory_allocator();

    GatherPerDeviceState state = {managed_handle.raw_handle(),
                                  legion_dim_t{0_n}};

    SUBCASE("forward_kernel") {
      auto run_forward_test = [&](TensorShape const &input_shape,
                                  TensorShape const &index_shape,
                                  TensorShape const &output_shape) {
        GenericTensorAccessorR input_accessor =
            create_random_filled_accessor_r(input_shape, allocator);
        GenericTensorAccessorR index_accessor =
            create_random_filled_accessor_r(index_shape, allocator);
        GenericTensorAccessorW output_accessor =
            allocator.allocate_tensor(output_shape);

        Kernels::Gather::forward_kernel(/*stream=*/managed_stream.raw_stream(),
                                        /*per_device_state=*/state,
                                        /*input=*/input_accessor,
                                        /*index=*/index_accessor,
                                        /*output=*/output_accessor);

        CHECK(contains_non_zero(output_accessor));
      };

      SUBCASE("test gather forward, 2D") {
        TensorShape input_shape =
            make_tensor_shape(FFOrdered{2_n, 100_n}, DataType::FLOAT);
        TensorShape index_shape =
            make_tensor_shape(FFOrdered{2_n, 20_n}, DataType::INT32);
        TensorShape output_shape =
            make_tensor_shape(FFOrdered{2_n, 20_n}, DataType::FLOAT);
        run_forward_test(input_shape, index_shape, output_shape);
      }

      SUBCASE("test gather forward, 1D") {
        TensorShape input_shape =
            make_tensor_shape(FFOrdered{100_n}, DataType::FLOAT);
        TensorShape index_shape =
            make_tensor_shape(FFOrdered{10_n}, DataType::INT32);
        TensorShape output_shape =
            make_tensor_shape(FFOrdered{10_n}, DataType::FLOAT);
        run_forward_test(input_shape, index_shape, output_shape);
      }
    }

    SUBCASE("backward_kernel") {
      auto run_backward_test = [&](TensorShape const &input_shape,
                                   TensorShape const &index_shape,
                                   TensorShape const &output_shape) {
        GenericTensorAccessorR output_grad_accessor =
            create_random_filled_accessor_r(output_shape, allocator);
        GenericTensorAccessorR index_accessor =
            create_random_filled_accessor_r(index_shape, allocator);
        GenericTensorAccessorW input_grad_accessor =
            allocator.allocate_tensor(input_shape);

        Kernels::Gather::backward_kernel(/*stream=*/managed_stream.raw_stream(),
                                         /*per_device_state=*/state,
                                         /*output_grad=*/output_grad_accessor,
                                         /*index=*/index_accessor,
                                         /*input_grad=*/input_grad_accessor);
        CHECK(contains_non_zero(input_grad_accessor));
      };

      SUBCASE("test gather backward, 2D") {
        TensorShape input_shape =
            make_tensor_shape(FFOrdered{2_n, 100_n}, DataType::FLOAT);
        TensorShape index_shape =
            make_tensor_shape(FFOrdered{2_n, 25_n}, DataType::INT32);
        TensorShape output_shape =
            make_tensor_shape(FFOrdered{2_n, 25_n}, DataType::FLOAT);
        run_backward_test(input_shape, index_shape, output_shape);
      }
    }
  }
}
