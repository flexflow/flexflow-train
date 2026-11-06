#include "kernels/batch_norm_kernels_gpu.h"
#include "kernels/accessors_are_approximately_equal.h"
#include "kernels/accessors_are_equal.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/create_random_filled_accessor.h"
#include "kernels/format_accessor_contents.h"
#include "kernels/local_cuda_allocator.h"
#include "kernels/managed_ff_stream.h"
#include "kernels/managed_per_device_ff_handle.h"
#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;
using namespace ::FlexFlow::Kernels::BatchNorm;

static BatchNormAttrs make_attrs() {
  return BatchNormAttrs{
      /*activation=*/std::nullopt,
      /*affine=*/true,
      /*eps=*/1e-5,
      /*momentum=*/0.1,
  };
}

// NCHW
static GenericTensorAccessorR make_input(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {{1, 2}, {3, 4}},
              {{-1, -2}, {0.5, 0.25}},
          },
          {
              {{5, 6}, {7, 8}},
              {{2, 0.125}, {-3, 1}},
          },
      },
      allocator);
}

static GenericTensorAccessorR make_gamma(Allocator &allocator) {
  return create_1d_accessor_r_with_contents<float>({2, 0.5}, allocator);
}

static GenericTensorAccessorR make_beta(Allocator &allocator) {
  return create_1d_accessor_r_with_contents<float>({-1, 3}, allocator);
}

static GenericTensorAccessorR make_output(Allocator &allocator) {
  return create_4d_accessor_r_with_contents<float>(
      {
          {
              {{-4.055047512054443, -3.1821770668029785},
               {-2.3093061447143555, -1.4364354610443115}},
              {{2.760241985321045, 2.433763027191162},
               {3.249960422515869, 3.1683406829833984}},
          },
          {
              {{-0.5635647177696228, 0.3093060255050659},
               {1.1821768283843994, 2.0550475120544434}},
              {{3.7396788597106934, 3.127530813217163},
               {2.1072840690612793, 3.4131999015808105}},
          },
      },
      allocator);
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("batch_norm_gpu_forward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    BatchNormAttrs attrs = make_attrs();

    GenericTensorAccessorR input = make_input(allocator);
    GenericTensorAccessorR gamma = make_gamma(allocator);
    GenericTensorAccessorR beta = make_beta(allocator);

    // Intentionally randomize this tensor so we can be confident we never read
    // it
    GenericTensorAccessorW output =
        create_random_filled_accessor_w(input.shape, allocator);

    BatchNormPerDeviceState per_device_state =
        gpu_init_kernel(managed_handle.raw_handle(),
                        allocator,
                        nullptr,
                        2,
                        2,
                        2,
                        2,
                        attrs.activation,
                        attrs.eps);

    gpu_forward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*per_device_state=*/per_device_state,
        /*input_ptr=*/input.get_float_ptr(),
        /*output_ptr=*/output.get_float_ptr(),
        /*scale_ptr=*/gamma.get_float_ptr(),
        /*bias_ptr=*/beta.get_float_ptr());

    GenericTensorAccessorR correct = make_output(allocator);

    CHECK_MESSAGE(accessors_are_approximately_equal(output, correct),
                  check_kv("output", format_accessor_w_contents(output)));

    gpu_cleanup_kernel(allocator, per_device_state);
  }

  TEST_CASE("batch_norm_gpu_backward_kernel") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};

    Allocator allocator = create_local_cuda_memory_allocator();

    BatchNormAttrs attrs = make_attrs();

    GenericTensorAccessorR input = make_input(allocator);
    GenericTensorAccessorR gamma = make_gamma(allocator);
    GenericTensorAccessorR beta = make_beta(allocator);

    GenericTensorAccessorW forward_output =
        create_random_filled_accessor_w(input.shape, allocator);

    BatchNormPerDeviceState per_device_state =
        gpu_init_kernel(managed_handle.raw_handle(),
                        allocator,
                        nullptr,
                        2,
                        2,
                        2,
                        2,
                        attrs.activation,
                        attrs.eps);

    // cudnnBatchNormalizationBackward reads the batch statistics saved by the
    // forward pass, so the forward kernel has to run first
    gpu_forward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*per_device_state=*/per_device_state,
        /*input_ptr=*/input.get_float_ptr(),
        /*output_ptr=*/forward_output.get_float_ptr(),
        /*scale_ptr=*/gamma.get_float_ptr(),
        /*bias_ptr=*/beta.get_float_ptr());

    GenericTensorAccessorR output = make_output(allocator);

    GenericTensorAccessorW output_grad =
        create_4d_accessor_w_with_contents<float>(
            {
                {
                    {{-0.875, -0.75}, {-0.625, -0.5}},
                    {{-0.375, -0.25}, {-0.125, 0}},
                },
                {
                    {{0.125, 0.25}, {0.375, 0.5}},
                    {{0.625, 0.75}, {0.875, 1}},
                },
            },
            allocator);

    // Initialize with nonzero values to verify replacement.
    GenericTensorAccessorW input_grad =
        create_4d_accessor_w_with_contents<float>(
            {
                {
                    {{-1.75, -1.5}, {-1.25, -1}},
                    {{-0.75, -0.5}, {-0.25, 0}},
                },
                {
                    {{0.25, 0.5}, {0.75, 1}},
                    {{1.25, 1.5}, {1.75, 2}},
                },
            },
            allocator);

    GenericTensorAccessorW gamma_grad =
        create_1d_accessor_w_with_contents<float>({7, -3}, allocator);

    GenericTensorAccessorW beta_grad =
        create_1d_accessor_w_with_contents<float>({0.5, 11}, allocator);

    gpu_backward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*per_device_state=*/per_device_state,
        /*output_ptr=*/output.get_float_ptr(),
        /*output_grad_ptr=*/output_grad.get_float_ptr(),
        /*input_ptr=*/input.get_float_ptr(),
        /*input_grad_ptr=*/input_grad.get_float_ptr(),
        /*scale_ptr=*/gamma.get_float_ptr(),
        /*scale_grad_ptr=*/gamma_grad.get_float_ptr(),
        /*bias_grad_ptr=*/beta_grad.get_float_ptr(),
        /*numElements=*/
        get_num_elements(input.shape.dims).int_from_positive_int());

    GenericTensorAccessorR correct_input_grad =
        create_4d_accessor_r_with_contents<float>(
            {
                {
                    {{0.0727379322052002, -0.01039218902587891},
                     {-0.09352242946624756, -0.1766525506973267}},
                    {{-0.2091808915138245, -0.1475732326507568},
                     {-0.1587578058242798, -0.1127485111355782}},
                },
                {
                    {{0.1766526103019714, 0.09352242946624756},
                     {0.01039218902587891, -0.07273799180984497}},
                    {{0.05490469932556152, 0.134710431098938},
                     {0.2405133247375488, 0.1981320381164551}},
                },
            },
            allocator);

    GenericTensorAccessorR correct_gamma_grad =
        create_1d_accessor_r_with_contents<float>(
            {4.037027359008789, 0.7804887294769287}, allocator);

    GenericTensorAccessorR correct_beta_grad =
        create_1d_accessor_r_with_contents<float>({-1.5, 2.5}, allocator);

    CHECK_MESSAGE(
        // cuDNN's batch norm data gradient is not bit-identical to PyTorch's
        accessors_are_approximately_equal(input_grad, correct_input_grad, 1e-4),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));

    CHECK_MESSAGE(
        accessors_are_approximately_equal(gamma_grad, correct_gamma_grad),
        check_kv("gamma_grad", format_accessor_w_contents(gamma_grad)));

    CHECK_MESSAGE(accessors_are_equal(beta_grad, correct_beta_grad),
                  check_kv("beta_grad", format_accessor_w_contents(beta_grad)));

    gpu_cleanup_kernel(allocator, per_device_state);
  }
}

TEST_SUITE(FF_CUDA_TEST_SUITE) {
  TEST_CASE("BatchNorm configurable epsilon and fused ReLU") {
    ManagedPerDeviceFFHandle handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream stream{};
    Allocator allocator = create_local_cuda_memory_allocator();
    std::optional<Activation> activation = std::nullopt;
    SUBCASE("without activation") {}
    SUBCASE("ReLU") {
      activation = Activation::RELU;
    }

    GenericTensorAccessorR input = create_4d_accessor_r_with_contents<float>(
        {{{{1}}}, {{{3}}}}, allocator);
    GenericTensorAccessorR scale =
        create_1d_accessor_r_with_contents<float>({2}, allocator);
    GenericTensorAccessorR bias =
        create_1d_accessor_r_with_contents<float>({0.5}, allocator);
    GenericTensorAccessorW output =
        create_random_filled_accessor_w(input.shape, allocator);
    BatchNormPerDeviceState state = gpu_init_kernel(
        handle.raw_handle(), allocator, nullptr, 2, 1, 1, 1, activation, 0.25f);

    TensorShape channel_shape{TensorDims{FFOrdered{1_p}}, DataType::FLOAT};
    GenericTensorAccessorR running_var{
        channel_shape, state.runningVar, DeviceType::GPU};
    GenericTensorAccessorR one =
        create_1d_accessor_r_with_contents<float>({1}, allocator);
    CHECK(accessors_are_equal(running_var, one));

    gpu_forward_kernel(stream.raw_stream(),
                       state,
                       input.get_float_ptr(),
                       output.get_float_ptr(),
                       scale.get_float_ptr(),
                       bias.get_float_ptr());
    GenericTensorAccessorR expected = create_4d_accessor_r_with_contents<float>(
        {{{{activation.has_value() ? 0.0f : -1.2888543819998317f}}},
         {{{2.2888543819998317f}}}},
        allocator);
    CHECK(accessors_are_approximately_equal(output, expected));

    GenericTensorAccessorW output_grad =
        create_4d_accessor_w_with_contents<float>({{{{1}}}, {{{2}}}},
                                                  allocator);
    GenericTensorAccessorW input_grad =
        create_random_filled_accessor_w(input.shape, allocator);
    GenericTensorAccessorW scale_grad =
        create_random_filled_accessor_w(channel_shape, allocator);
    GenericTensorAccessorW bias_grad =
        create_random_filled_accessor_w(channel_shape, allocator);
    gpu_backward_kernel(stream.raw_stream(),
                        state,
                        output.get_float_ptr(),
                        output_grad.get_float_ptr(),
                        input.get_float_ptr(),
                        input_grad.get_float_ptr(),
                        scale.get_float_ptr(),
                        scale_grad.get_float_ptr(),
                        bias_grad.get_float_ptr(),
                        2);

    float dx =
        activation.has_value() ? 0.35777087639996635f : 0.17888543819998318f;
    GenericTensorAccessorR expected_input_grad =
        create_4d_accessor_r_with_contents<float>({{{{-dx}}}, {{{dx}}}},
                                                  allocator);
    GenericTensorAccessorR expected_scale_grad =
        create_1d_accessor_r_with_contents<float>({activation.has_value()
                                                       ? 1.7888543819998317f
                                                       : 0.8944271909999159f},
                                                  allocator);
    GenericTensorAccessorR expected_bias_grad =
        create_1d_accessor_r_with_contents<float>(
            {activation.has_value() ? 2.0f : 3.0f}, allocator);
    CHECK(accessors_are_approximately_equal(input_grad, expected_input_grad));
    CHECK(accessors_are_approximately_equal(scale_grad, expected_scale_grad));
    CHECK(accessors_are_equal(bias_grad, expected_bias_grad));
    gpu_cleanup_kernel(allocator, state);
  }
}
