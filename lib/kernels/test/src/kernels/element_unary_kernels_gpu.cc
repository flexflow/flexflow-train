#include "kernels/element_unary_kernels_gpu.h"
#include "kernels/accessors_are_approximately_equal.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "op-attrs/ops/element_unary.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static ElementUnaryAttrs make_silu_attrs() {
  return ElementUnaryAttrs{
      /*op=*/ElementUnaryOp::SILU,
      /*scalar=*/std::nullopt,
  };
}

static GenericTensorAccessorR make_input(Allocator &allocator) {
  return create_2d_accessor_r_with_contents<float>(
      {
          {3, -3, 6},
          {0, 1, 5},
          {1, 2, -2},
          {-8, 0.5, -3},
      },
      allocator);
}

static GenericTensorAccessorR make_output_grad(Allocator &allocator) {
  return create_2d_accessor_r_with_contents<float>(
      {
          {1, 2, -1},
          {6, 4, 2},
          {0.5, 0.1, -2},
          {0, 0.5, 0},
      },
      allocator);
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("element_unary_gpu_forward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    GenericTensorAccessorR input = make_input(allocator);

    SUBCASE("relu") {
      ElementUnaryAttrs attrs = make_relu_attrs();

      // Intentionally randomize this tensor so we can be confident we never
      // read it
      GenericTensorAccessorW result =
          create_random_filled_accessor_w(input.shape, allocator);

      ElementUnaryPerDeviceState per_device_state =
          element_unary_gpu_init_kernel(attrs, input.shape, result.shape);

      element_unary_gpu_forward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/result);

      GenericTensorAccessorR correct =
          create_2d_accessor_r_with_contents<float>(
              {
                  {3, 0, 6},
                  {0, 1, 5},
                  {1, 2, 0},
                  {0, 0.5, 0},
              },
              allocator);

      CHECK_MESSAGE(accessors_are_equal(result, correct),
                    check_kv("result", format_accessor_w_contents(result)));
    }

    SUBCASE("silu") {
      ElementUnaryAttrs attrs = make_silu_attrs();

      // Intentionally randomize this tensor so we can be confident we never
      // read it
      GenericTensorAccessorW result =
          create_random_filled_accessor_w(input.shape, allocator);

      ElementUnaryPerDeviceState per_device_state =
          element_unary_gpu_init_kernel(attrs, input.shape, result.shape);

      element_unary_gpu_forward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*input=*/input,
          /*output=*/result);

      GenericTensorAccessorR correct = create_2d_accessor_r_with_contents<
          float>(
          {
              {2.857722520828247, -0.14227761328220367, 5.985164642333984},
              {0.0, 0.7310585975646973, 4.966535568237305},
              {0.7310585975646973, 1.7615940570831299, -0.23840583860874176},
              {-0.00268280110321939, 0.3112296760082245, -0.14227761328220367},
          },
          allocator);

      // CUDA's expf is not bit-identical to PyTorch's CPU implementation
      CHECK_MESSAGE(accessors_are_approximately_equal(result, correct, 1e-6),
                    check_kv("result", format_accessor_w_contents(result)));
    }
  }

  TEST_CASE("element_unary_gpu_backward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    GenericTensorAccessorR input = make_input(allocator);
    GenericTensorAccessorR output_grad = make_output_grad(allocator);

    // Initialize with nonzero values to verify replacement.
    auto make_input_grad = [&]() {
      return create_2d_accessor_w_with_contents<float>(
          {
              {10, 11, 12},
              {13, 14, 15},
              {16, 17, 18},
              {19, 20, 21},
          },
          allocator);
    };

    SUBCASE("relu") {
      ElementUnaryAttrs attrs = make_relu_attrs();

      GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
          {
              {3, 0, 6},
              {0, 1, 5},
              {1, 2, 0},
              {0, 0.5, 0},
          },
          allocator);

      GenericTensorAccessorW input_grad = make_input_grad();

      ElementUnaryPerDeviceState per_device_state =
          element_unary_gpu_init_kernel(attrs, input.shape, output.shape);

      element_unary_gpu_backward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*input=*/input,
          /*input_grad=*/input_grad);

      GenericTensorAccessorR correct_input_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1, 0, -1},
                  {0, 4, 2},
                  {0.5, 0.1, 0},
                  {0, 0.5, 0},
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_equal(input_grad, correct_input_grad),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }

    SUBCASE("silu") {
      ElementUnaryAttrs attrs = make_silu_attrs();

      GenericTensorAccessorR output = create_2d_accessor_r_with_contents<float>(
          {
              {2.857722520828247, -0.14227761328220367, 5.985164642333984},
              {0.0, 0.7310585975646973, 4.966535568237305},
              {0.7310585975646973, 1.7615940570831299, -0.23840583860874176},
              {-0.00268280110321939, 0.3112296760082245, -0.14227761328220367},
          },
          allocator);

      GenericTensorAccessorW input_grad = make_input_grad();

      ElementUnaryPerDeviceState per_device_state =
          element_unary_gpu_init_kernel(attrs, input.shape, output.shape);

      element_unary_gpu_backward_kernel(
          /*stream=*/managed_stream.raw_stream(),
          /*handle=*/managed_handle.raw_handle(),
          /*per_device_state=*/per_device_state,
          /*attrs=*/attrs,
          /*output=*/output,
          /*output_grad=*/output_grad,
          /*input=*/input,
          /*input_grad=*/input_grad);

      GenericTensorAccessorR correct_input_grad =
          create_2d_accessor_r_with_contents<float>(
              {
                  {1.088104106015169, -0.1762082120303393, -1.012326432591525},
                  {3, 3.710682047485947, 2.053094864859331},
                  {0.4638352559357434, 0.1090784248784896, 0.1815684975697909},
                  {-0, 0.3699805936513259, -0},
              },
              allocator);

      CHECK_MESSAGE(
          accessors_are_approximately_equal(
              input_grad, correct_input_grad, 1e-5),
          check_kv("input_grad", format_accessor_w_contents(input_grad)));
    }
  }
}
