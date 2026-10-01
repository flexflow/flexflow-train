#include "kernels/batch_norm_kernels_gpu.h"
#include "internal/test_utils.h"
#include "kernels/create_accessor_with_contents.h"
#include "kernels/element_unary_kernels_gpu.h"
#include "kernels/format_accessor_contents.h"
#include "test/utils/doctest/check_kv.h"
#include <doctest/doctest.h>

using namespace ::FlexFlow;

static BatchNormAttrs
    make_attrs(std::optional<Activation> activation = std::nullopt) {
  return BatchNormAttrs{
      /*activation=*/activation,
      /*affine=*/true,
      /*eps=*/1e-5,
      /*momentum=*/0.1,
      /*mode=*/BatchNormMode::SPATIAL_PERSISTENT,
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
        batch_norm_gpu_init_kernel(allocator, attrs, input.shape, output.shape);

    batch_norm_gpu_forward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*handle=*/managed_handle.raw_handle(),
        /*per_device_state=*/per_device_state,
        /*attrs=*/attrs,
        /*input=*/input,
        /*gamma=*/gamma,
        /*beta=*/beta,
        /*output=*/output);

    GenericTensorAccessorR correct = make_output(allocator);

    CHECK_MESSAGE(accessors_are_equal(output, correct),
                  check_kv("output", format_accessor_w_contents(output)));

    batch_norm_gpu_cleanup_kernel(allocator, per_device_state);
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

    BatchNormPerDeviceState per_device_state = batch_norm_gpu_init_kernel(
        allocator, attrs, input.shape, forward_output.shape);

    // cudnnBatchNormalizationBackward reads the batch statistics saved by the
    // forward pass, so the forward kernel has to run first
    batch_norm_gpu_forward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*handle=*/managed_handle.raw_handle(),
        /*per_device_state=*/per_device_state,
        /*attrs=*/attrs,
        /*input=*/input,
        /*gamma=*/gamma,
        /*beta=*/beta,
        /*output=*/forward_output);

    GenericTensorAccessorR output = make_output(allocator);

    GenericTensorAccessorR output_grad =
        create_4d_accessor_r_with_contents<float>(
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

    // The gradients are accumulated into, so they need to start from known
    // values
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

    batch_norm_gpu_backward_kernel(
        /*stream=*/managed_stream.raw_stream(),
        /*handle=*/managed_handle.raw_handle(),
        /*per_device_state=*/per_device_state,
        /*attrs=*/attrs,
        /*output=*/output,
        /*output_grad=*/output_grad,
        /*input=*/input,
        /*input_grad=*/input_grad,
        /*gamma=*/gamma,
        /*beta=*/beta,
        /*gamma_grad=*/gamma_grad,
        /*beta_grad=*/beta_grad);

    GenericTensorAccessorR correct_input_grad =
        create_4d_accessor_r_with_contents<float>(
            {
                {
                    {{-1.6772620677947998, -1.510392189025879},
                     {-1.3435224294662476, -1.1766525506973267}},
                    {{-0.9591808915138245, -0.6475732326507568},
                     {-0.4087578058242798, -0.11274851113557816}},
                },
                {
                    {{0.42665261030197144, 0.5935224294662476},
                     {0.7603921890258789, 0.927262008190155}},
                    {{1.3049046993255615, 1.634710431098938},
                     {1.9905133247375488, 2.198132038116455}},
                },
            },
            allocator);

    GenericTensorAccessorR correct_gamma_grad =
        create_1d_accessor_r_with_contents<float>(
            {11.037027359008789, -2.2195112705230713}, allocator);

    GenericTensorAccessorR correct_beta_grad =
        create_1d_accessor_r_with_contents<float>({-1.0, 13.5}, allocator);

    CHECK_MESSAGE(
        // cuDNN's batch norm data gradient is not bit-identical to PyTorch's
        accessors_within_epsilon(input_grad, correct_input_grad, 1e-6),
        check_kv("input_grad", format_accessor_w_contents(input_grad)));

    CHECK_MESSAGE(
        accessors_are_equal(gamma_grad, correct_gamma_grad),
        check_kv("gamma_grad", format_accessor_w_contents(gamma_grad)));

    CHECK_MESSAGE(accessors_are_equal(beta_grad, correct_beta_grad),
                  check_kv("beta_grad", format_accessor_w_contents(beta_grad)));

    batch_norm_gpu_cleanup_kernel(allocator, per_device_state);
  }

  // Applying an activation as part of batch norm has to compute what the two
  // separate operators it replaces would have computed. Comparing against those
  // operators rather than against a table of expected numbers is the point:
  // what matters is that folding them together did not change the answer.
  TEST_CASE("batch_norm_gpu fused activation matches the separate operators") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};
    Allocator allocator = create_local_cuda_memory_allocator();

    ElementUnaryAttrs silu_attrs = ElementUnaryAttrs{
        /*op_type=*/OperatorType::SILU,
        /*scalar=*/std::nullopt,
    };

    GenericTensorAccessorR input = make_input(allocator);
    GenericTensorAccessorR gamma = make_gamma(allocator);
    GenericTensorAccessorR beta = make_beta(allocator);

    ElementUnaryPerDeviceState silu_state =
        element_unary_gpu_init_kernel(silu_attrs, input.shape, input.shape);

    SUBCASE("forward") {
      BatchNormAttrs separate_attrs = make_attrs();
      BatchNormAttrs fused_attrs = make_attrs(Activation::SILU);

      GenericTensorAccessorW normalized =
          create_random_filled_accessor_w(input.shape, allocator);
      GenericTensorAccessorW separate =
          create_random_filled_accessor_w(input.shape, allocator);
      GenericTensorAccessorW fused =
          create_random_filled_accessor_w(input.shape, allocator);

      BatchNormPerDeviceState separate_state = batch_norm_gpu_init_kernel(
          allocator, separate_attrs, input.shape, input.shape);
      batch_norm_gpu_forward_kernel(managed_stream.raw_stream(),
                                    managed_handle.raw_handle(),
                                    separate_state,
                                    separate_attrs,
                                    input,
                                    gamma,
                                    beta,
                                    normalized);
      element_unary_gpu_forward_kernel(
          managed_stream.raw_stream(),
          managed_handle.raw_handle(),
          silu_state,
          silu_attrs,
          read_only_accessor_from_write_accessor(normalized),
          separate);

      BatchNormPerDeviceState fused_state = batch_norm_gpu_init_kernel(
          allocator, fused_attrs, input.shape, input.shape);
      batch_norm_gpu_forward_kernel(managed_stream.raw_stream(),
                                    managed_handle.raw_handle(),
                                    fused_state,
                                    fused_attrs,
                                    input,
                                    gamma,
                                    beta,
                                    fused);

      CHECK_MESSAGE(accessors_within_epsilon(fused, separate, 1e-5),
                    check_kv("fused", format_accessor_w_contents(fused)),
                    check_kv("separate", format_accessor_w_contents(separate)));

      batch_norm_gpu_cleanup_kernel(allocator, separate_state);
      batch_norm_gpu_cleanup_kernel(allocator, fused_state);
    }

    SUBCASE("backward") {
      BatchNormAttrs separate_attrs = make_attrs();
      BatchNormAttrs fused_attrs = make_attrs(Activation::SILU);

      GenericTensorAccessorR output_grad =
          create_random_filled_accessor_r(input.shape, allocator);

      // The backward kernels accumulate into their gradients, so every
      // gradient starts at zero for the two paths to be comparable.

      // The separate path: batch norm writes the value the activation reads,
      // and the activation's backward turns the output gradient into the
      // gradient of that value.
      GenericTensorAccessorW normalized =
          create_random_filled_accessor_w(input.shape, allocator);
      GenericTensorAccessorW activated =
          create_random_filled_accessor_w(input.shape, allocator);
      GenericTensorAccessorW normalized_grad =
          create_zero_filled_accessor_w(input.shape, allocator);
      GenericTensorAccessorW separate_input_grad =
          create_zero_filled_accessor_w(input.shape, allocator);
      GenericTensorAccessorW separate_gamma_grad =
          create_zero_filled_accessor_w(gamma.shape, allocator);
      GenericTensorAccessorW separate_beta_grad =
          create_zero_filled_accessor_w(gamma.shape, allocator);

      BatchNormPerDeviceState separate_state = batch_norm_gpu_init_kernel(
          allocator, separate_attrs, input.shape, input.shape);
      batch_norm_gpu_forward_kernel(managed_stream.raw_stream(),
                                    managed_handle.raw_handle(),
                                    separate_state,
                                    separate_attrs,
                                    input,
                                    gamma,
                                    beta,
                                    normalized);
      element_unary_gpu_forward_kernel(
          managed_stream.raw_stream(),
          managed_handle.raw_handle(),
          silu_state,
          silu_attrs,
          read_only_accessor_from_write_accessor(normalized),
          activated);
      element_unary_gpu_backward_kernel(
          managed_stream.raw_stream(),
          managed_handle.raw_handle(),
          silu_state,
          silu_attrs,
          read_only_accessor_from_write_accessor(activated),
          output_grad,
          read_only_accessor_from_write_accessor(normalized),
          normalized_grad);
      batch_norm_gpu_backward_kernel(
          managed_stream.raw_stream(),
          managed_handle.raw_handle(),
          separate_state,
          separate_attrs,
          read_only_accessor_from_write_accessor(normalized),
          read_only_accessor_from_write_accessor(normalized_grad),
          input,
          separate_input_grad,
          gamma,
          beta,
          separate_gamma_grad,
          separate_beta_grad);

      // The fused path, which never writes the value between them.
      GenericTensorAccessorW fused_output =
          create_random_filled_accessor_w(input.shape, allocator);
      GenericTensorAccessorW fused_input_grad =
          create_zero_filled_accessor_w(input.shape, allocator);
      GenericTensorAccessorW fused_gamma_grad =
          create_zero_filled_accessor_w(gamma.shape, allocator);
      GenericTensorAccessorW fused_beta_grad =
          create_zero_filled_accessor_w(gamma.shape, allocator);

      BatchNormPerDeviceState fused_state = batch_norm_gpu_init_kernel(
          allocator, fused_attrs, input.shape, input.shape);
      batch_norm_gpu_forward_kernel(managed_stream.raw_stream(),
                                    managed_handle.raw_handle(),
                                    fused_state,
                                    fused_attrs,
                                    input,
                                    gamma,
                                    beta,
                                    fused_output);
      batch_norm_gpu_backward_kernel(
          managed_stream.raw_stream(),
          managed_handle.raw_handle(),
          fused_state,
          fused_attrs,
          read_only_accessor_from_write_accessor(fused_output),
          output_grad,
          input,
          fused_input_grad,
          gamma,
          beta,
          fused_gamma_grad,
          fused_beta_grad);

      CHECK_MESSAGE(
          accessors_within_epsilon(fused_input_grad, separate_input_grad, 1e-5),
          check_kv("fused", format_accessor_w_contents(fused_input_grad)),
          check_kv("separate",
                   format_accessor_w_contents(separate_input_grad)));
      CHECK_MESSAGE(
          accessors_within_epsilon(fused_gamma_grad, separate_gamma_grad, 1e-5),
          check_kv("fused", format_accessor_w_contents(fused_gamma_grad)));
      CHECK_MESSAGE(
          accessors_within_epsilon(fused_beta_grad, separate_beta_grad, 1e-5),
          check_kv("fused", format_accessor_w_contents(fused_beta_grad)));

      batch_norm_gpu_cleanup_kernel(allocator, separate_state);
      batch_norm_gpu_cleanup_kernel(allocator, fused_state);
    }
  }

  // The case above is too small to reach the vectorized walk or the split
  // kernels, which only run once the spatial size (and, for the split, a
  // channel count too small to fill the device) makes them worth it. Here the
  // gradients also start from the same non-zero values on both paths, since
  // the fused kernels have to accumulate into them just as cuDNN does.
  TEST_CASE("batch_norm_gpu fused activation matches the separate operators at "
            "sizes that reach the vectorized and split kernels") {
    ManagedPerDeviceFFHandle managed_handle = initialize_single_gpu_handle(
        /*workSpaceSize=*/1024 * 1024,
        /*allowTensorOpMathConversion=*/true);
    ManagedFFStream managed_stream{};
    Allocator allocator = create_local_cuda_memory_allocator();

    ffStream_t stream = managed_stream.raw_stream();
    PerDeviceFFHandle handle = managed_handle.raw_handle();

    ElementUnaryAttrs silu_attrs = ElementUnaryAttrs{
        /*op_type=*/OperatorType::SILU,
        /*scalar=*/std::nullopt,
    };
    BatchNormAttrs separate_attrs = make_attrs();
    BatchNormAttrs fused_attrs = make_attrs(Activation::SILU);

    auto check = [&](TensorShape const &input_shape) {
      TensorShape channel_shape = TensorShape{
          TensorDims{FFOrdered{dim_at_idx(input_shape.dims, ff_dim_t{1_n})}},
          DataType::FLOAT,
      };

      GenericTensorAccessorR input =
          create_random_filled_accessor_r(input_shape, allocator);
      GenericTensorAccessorR gamma =
          create_random_filled_accessor_r(channel_shape, allocator);
      GenericTensorAccessorR beta =
          create_random_filled_accessor_r(channel_shape, allocator);
      GenericTensorAccessorR output_grad =
          create_random_filled_accessor_r(input_shape, allocator);

      auto copy_of = [&](GenericTensorAccessorW const &accessor) {
        GenericTensorAccessorW result =
            allocator.allocate_tensor(accessor.shape);
        copy_accessor_data_to_l_from_r(
            result, read_only_accessor_from_write_accessor(accessor));
        return result;
      };

      ElementUnaryPerDeviceState silu_state =
          element_unary_gpu_init_kernel(silu_attrs, input_shape, input_shape);

      // The separate path.
      GenericTensorAccessorW normalized =
          create_random_filled_accessor_w(input_shape, allocator);
      GenericTensorAccessorW separate_output =
          create_random_filled_accessor_w(input_shape, allocator);
      GenericTensorAccessorW normalized_grad =
          create_zero_filled_accessor_w(input_shape, allocator);
      GenericTensorAccessorW separate_input_grad =
          create_random_filled_accessor_w(input_shape, allocator);
      GenericTensorAccessorW separate_gamma_grad =
          create_random_filled_accessor_w(channel_shape, allocator);
      GenericTensorAccessorW separate_beta_grad =
          create_random_filled_accessor_w(channel_shape, allocator);

      // The fused path, starting from the same gradients.
      GenericTensorAccessorW fused_output =
          create_random_filled_accessor_w(input_shape, allocator);
      GenericTensorAccessorW fused_input_grad = copy_of(separate_input_grad);
      GenericTensorAccessorW fused_gamma_grad = copy_of(separate_gamma_grad);
      GenericTensorAccessorW fused_beta_grad = copy_of(separate_beta_grad);

      BatchNormPerDeviceState separate_state = batch_norm_gpu_init_kernel(
          allocator, separate_attrs, input_shape, input_shape);
      batch_norm_gpu_forward_kernel(stream,
                                    handle,
                                    separate_state,
                                    separate_attrs,
                                    input,
                                    gamma,
                                    beta,
                                    normalized);
      element_unary_gpu_forward_kernel(
          stream,
          handle,
          silu_state,
          silu_attrs,
          read_only_accessor_from_write_accessor(normalized),
          separate_output);
      element_unary_gpu_backward_kernel(
          stream,
          handle,
          silu_state,
          silu_attrs,
          read_only_accessor_from_write_accessor(separate_output),
          output_grad,
          read_only_accessor_from_write_accessor(normalized),
          normalized_grad);
      batch_norm_gpu_backward_kernel(
          stream,
          handle,
          separate_state,
          separate_attrs,
          read_only_accessor_from_write_accessor(normalized),
          read_only_accessor_from_write_accessor(normalized_grad),
          input,
          separate_input_grad,
          gamma,
          beta,
          separate_gamma_grad,
          separate_beta_grad);

      BatchNormPerDeviceState fused_state = batch_norm_gpu_init_kernel(
          allocator, fused_attrs, input_shape, input_shape);
      batch_norm_gpu_forward_kernel(stream,
                                    handle,
                                    fused_state,
                                    fused_attrs,
                                    input,
                                    gamma,
                                    beta,
                                    fused_output);
      batch_norm_gpu_backward_kernel(
          stream,
          handle,
          fused_state,
          fused_attrs,
          read_only_accessor_from_write_accessor(fused_output),
          output_grad,
          input,
          fused_input_grad,
          gamma,
          beta,
          fused_gamma_grad,
          fused_beta_grad);

      CHECK(accessors_within_epsilon(fused_output, separate_output, 1e-4));
      CHECK(accessors_within_epsilon(
          fused_input_grad, separate_input_grad, 1e-4));
      CHECK_MESSAGE(
          accessors_within_epsilon(fused_gamma_grad, separate_gamma_grad, 1e-3),
          check_kv("fused", format_accessor_w_contents(fused_gamma_grad)),
          check_kv("separate",
                   format_accessor_w_contents(separate_gamma_grad)));
      CHECK_MESSAGE(
          accessors_within_epsilon(fused_beta_grad, separate_beta_grad, 1e-3),
          check_kv("fused", format_accessor_w_contents(fused_beta_grad)),
          check_kv("separate", format_accessor_w_contents(separate_beta_grad)));

      batch_norm_gpu_cleanup_kernel(allocator, separate_state);
      batch_norm_gpu_cleanup_kernel(allocator, fused_state);
    };

    SUBCASE("vectorized") {
      // Enough channels to fill any current device, so not split.
      check(TensorShape{
          TensorDims{FFOrdered{2_p, 512_p, 32_p, 32_p}},
          DataType::FLOAT,
      });
    }

    SUBCASE("split") {
      check(TensorShape{
          TensorDims{FFOrdered{2_p, 4_p, 40_p, 40_p}},
          DataType::FLOAT,
      });
    }
  }
}
