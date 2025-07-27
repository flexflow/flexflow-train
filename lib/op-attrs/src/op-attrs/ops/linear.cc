#include "op-attrs/ops/linear.h"
#include "op-attrs/ff_ordered/slice.h"
#include "op-attrs/ff_ordered/transform.h"
#include "op-attrs/initializers/kaiming_initializer_mode.h"
#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/relative_ff_dim_t.h"
#include "op-attrs/tensor_dims.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/product.h"
#include "utils/expected.h"
#include "utils/fmt/optional.h"
#include "utils/integer_conversions.h"
#include "utils/orthotope/down_projection.h"
#include "utils/orthotope/eq_projection.h"
#include "utils/orthotope/up_projection.h"

namespace FlexFlow {

std::vector<IncomingTensorRole>
    get_linear_incoming_tensor_roles(LinearAttrs const &attrs) {
  std::vector<IncomingTensorRole> result = {
      IncomingTensorRole::INPUT,
      IncomingTensorRole::WEIGHT,
  };

  if (attrs.use_bias) {
    result.push_back(IncomingTensorRole::WEIGHT);
  }

  return result;
}

RecordFormatter as_dot(LinearAttrs const &attrs) {
  RecordFormatter r;

  auto kv = [](std::string const &label, auto const &val) {
    RecordFormatter rr;
    rr << label << fmt::to_string(val);
    return rr;
  };

  r << kv("out_channels", attrs.out_channels) << kv("use_bias", attrs.use_bias)
    << kv("data_type", attrs.data_type) << kv("activation", attrs.activation)
    << kv("regularizer", attrs.regularizer);

  return r;
}

tl::expected<TensorShape, std::string>
    get_projection_shape(LinearAttrs const &attrs,
                         TensorShape const &input_shape) {
  positive_int in_channels =
      dim_at_idx(input_shape.dims, relative_ff_dim_t{-1});

  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{attrs.out_channels, in_channels},
      },
      input_shape.data_type,
  };
}

tl::expected<TensorShape, std::string>
    get_bias_shape(LinearAttrs const &attrs, TensorShape const &input_shape) {
  return TensorShape{
      TensorDims{
          FFOrdered<positive_int>{attrs.out_channels},
      },
      input_shape.data_type,
  };
}

tl::expected<TensorShape, std::string>
    get_output_shape(LinearAttrs const &attrs, TensorShape const &input_shape) {
  TensorShape output_shape = input_shape;
  output_shape.dims.ff_ordered.at(relative_ff_dim_t{-1}) = attrs.out_channels;

  return output_shape;
}

tl::expected<std::vector<TensorShape>, std::string>
    get_weight_shapes(LinearAttrs const &attrs,
                      TensorShape const &input_shape) {

  std::vector<TensorShape> weight_shapes = {
      PROPAGATE_ERR(get_projection_shape(attrs, input_shape)),
  };

  if (attrs.use_bias) {
    weight_shapes.push_back(PROPAGATE_ERR(get_bias_shape(attrs, input_shape)));
  }

  return weight_shapes;
}

tl::expected<ParallelTensorShape, std::string>
    get_projection_shape(LinearAttrs const &attrs,
                         ParallelTensorShape const &input) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_projection_shape(attrs, get_reduced_shape(input));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  SumDegree sum_degree = SumDegree{1_p};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      get_sum_degree(input) * product(slice(ff_ordered_shard_degrees(input),
                                            relative_ff_dim_t{0},
                                            relative_ff_dim_t{-1}))};
  FFOrdered<positive_int> shard_degrees = FFOrdered<positive_int>{
      get_discard_copy_degree(input),
      shard_dim_at_idx(input, relative_ff_dim_t{-1}).degree,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_bias_shape(LinearAttrs const &attrs, ParallelTensorShape const &input) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_bias_shape(attrs, get_reduced_shape(input));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  SumDegree sum_degree =
      SumDegree{get_sum_degree(input) *
                shard_dim_at_idx(input, relative_ff_dim_t{-1}).degree};
  DiscardCopyDegree discard_copy_degree =
      DiscardCopyDegree{product(slice(ff_ordered_shard_degrees(input),
                                      relative_ff_dim_t{0},
                                      relative_ff_dim_t{-1}))};
  FFOrdered<positive_int> shard_degrees =
      FFOrdered<positive_int>{get_discard_copy_degree(input)};

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(LinearAttrs const &attrs,
                     ParallelTensorShape const &input) {
  TensorShape unpar = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_output_shape(attrs, get_reduced_shape(input));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }
    result_unpar.value();
  });

  SumDegree sum_degree =
      SumDegree{get_sum_degree(input) *
                shard_dim_at_idx(input, relative_ff_dim_t{-1}).degree};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1_p};
  FFOrdered<positive_int> shard_degrees = ff_ordered_shard_degrees(input);
  shard_degrees.at(relative_ff_dim_t{-1}) = get_discard_copy_degree(input);

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<std::vector<ParallelTensorShape>, std::string>
    get_weight_shapes(LinearAttrs const &attrs,
                      ParallelTensorShape const &input_shape) {

  std::vector<ParallelTensorShape> weight_shapes = {
      PROPAGATE_ERR(get_projection_shape(attrs, input_shape)),
  };

  if (attrs.use_bias) {
    weight_shapes.push_back(PROPAGATE_ERR(get_bias_shape(attrs, input_shape)));
  }

  return weight_shapes;
}

/**
 * @brief Chosen to match pytorch implementation
 *
 * see
 * https://github.com/pytorch/pytorch/blob/1eba9b3aa3c43f86f4a2c807ac8e12c4a7767340/torch/nn/modules/linear.py#L114-L122
 */
tl::expected<std::vector<InitializerAttrs>, std::string> get_initializers(
    LinearAttrs const &attrs,
    TensorShape const &input_shape,
    std::optional<InitializerAttrs> const &maybe_projection_initializer,
    std::optional<InitializerAttrs> const &maybe_bias_initializer) {

  if (!attrs.use_bias && maybe_bias_initializer.has_value()) {
    return tl::unexpected(
        fmt::format("Expected bias_initializer=std::nullopt since "
                    "use_bias=false, but received bias_initializer: {}",
                    maybe_bias_initializer.value()));
  }

  TensorShape projection_shape =
      PROPAGATE_ERR(get_projection_shape(attrs, input_shape));

  InitializerAttrs projection_default_initializer =
      InitializerAttrs{KaimingNormalAttrs{
          /*a=*/sqrtf(5.0),
          /*mode=*/KaimingInitializerMode::FAN_IN,
          /*nonlinearity=*/KaimingInitializerNonlinearity::LEAKY_RELU,
          /*seed=*/0,
      }};

  InitializerAttrs projection_initializer =
      maybe_projection_initializer.value_or(projection_default_initializer);

  positive_int fan_in = calculate_fan_for_mode(projection_shape.dims,
                                               KaimingInitializerMode::FAN_IN);

  float bound = 1 / sqrtf(static_cast<float>(fan_in.int_from_positive_int()));

  InitializerAttrs bias_default_initializer =
      InitializerAttrs{UniformInitializerAttrs{
          /*seed=*/0,
          /*min_val=*/-bound,
          /*max_val=*/bound,
      }};

  InitializerAttrs bias_initializer =
      maybe_bias_initializer.value_or(bias_default_initializer);

  if (attrs.use_bias) {
    return std::vector{projection_initializer, bias_initializer};
  } else {
    return std::vector{projection_initializer};
  }
}

tl::expected<ParallelTensorSpaceMapping, std::string>
    get_input_to_output_projection(LinearAttrs const &attrs, nonnegative_int input_num_dims) {
  
  DownProjection<
    parallel_tensor_dim_idx_t, 
    parallel_tensor_dim_idx_t
  > inp_to_out = make_empty_down_projection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>();

  ff_dim_t input_channel_dim = ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t{-1}, input_num_dims);

  nonnegative_int output_num_dims = input_num_dims;
  ff_dim_t output_channel_dim = ff_dim_t_from_relative_ff_dim_t(relative_ff_dim_t{-1}, output_num_dims);

  project_dims(inp_to_out, 
               /*from=*/{sum_dim_idx(), shard_dim_idx(input_channel_dim)},
               /*onto=*/sum_dim_idx());
  project_dims(inp_to_out, 
               /*from=*/{discard_copy_dim_idx()}, 
               /*onto=*/shard_dim_idx(output_channel_dim));

  for (ff_dim_t const &idx : ff_dim_range(nonnegative_int{input_num_dims.unwrap_nonnegative() - 1})) {
    project_dims(inp_to_out, 
                 /*from=*/{shard_dim_idx(idx)}, 
                 /*onto=*/shard_dim_idx(idx));
  }

  return ParallelTensorSpaceMapping{DimProjection{inp_to_out}};
}

tl::expected<OperatorSpaceToParallelTensorSpaceMapping, std::string>
  get_operator_to_input_projection(LinearAttrs const &attrs,
                                   nonnegative_int input_num_dims) {

    nonnegative_int output_num_dims = input_num_dims;

    UpProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t> 
      out_to_inp = invert_down_projection(throw_if_unexpected(get_input_to_output_projection(attrs, input_num_dims)).raw_projection.require_down_proj());

    EqProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t> 
      op_to_out = throw_if_unexpected(get_operator_to_output_mapping(attrs, input_num_dims)).raw_projection.require_eq_proj();

    return OperatorSpaceToParallelTensorSpaceMapping{
      DimProjection{
        compose_up_projections(up_from_eq_proj(op_to_out), out_to_inp),
      },
    };
}

tl::expected<OperatorSpaceToParallelTensorSpaceMapping, std::string>
    get_operator_to_output_mapping(LinearAttrs const &attrs,
                                      nonnegative_int input_num_shard_dims) {
  nonnegative_int output_num_shard_dims = input_num_shard_dims;

  return get_identity_mapping(output_num_shard_dims);
}

} // namespace FlexFlow
