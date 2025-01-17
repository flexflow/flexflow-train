#include "op-attrs/ops/linear.h"
#include "op-attrs/dim_ordered/slice.h"
#include "op-attrs/dim_ordered/transform.h"
#include "op-attrs/ff_dim_t.h"
#include "op-attrs/operator_space_parallel_tensor_space_mapping.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_num_dims.dtg.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/product.h"
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
  size_t in_channels = dim_at_idx(input_shape, relative_ff_dim_t{-1});

  return TensorShape{
      TensorDims{
          FFOrdered<size_t>{in_channels, size_t_from_int(attrs.out_channels)},
      },
      input_shape.data_type,
  };
}

tl::expected<TensorShape, std::string>
    get_bias_shape(LinearAttrs const &attrs, TensorShape const &input_shape) {
  return TensorShape{
      TensorDims{
          FFOrdered<size_t>{size_t_from_int(attrs.out_channels)},
      },
      input_shape.data_type,
  };
}

tl::expected<TensorShape, std::string>
    get_output_shape(LinearAttrs const &attrs, TensorShape const &input_shape) {
  TensorShape output_shape = input_shape;
  output_shape.dims.ff_ordered.at(relative_ff_dim_t{-1}) =
      size_t_from_int(attrs.out_channels);

  return output_shape;
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

  SumDegree sum_degree = SumDegree{1};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      get_sum_degree(input) * product(slice(ff_ordered_shard_degrees(input),
                                            std::nullopt,
                                            relative_ff_dim_t{-1}))};
  FFOrdered<int> shard_degrees = FFOrdered<int>{
      shard_dim_at_idx(input, relative_ff_dim_t{-1}).degree,
      get_discard_copy_degree(input),
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
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{product(slice(
      ff_ordered_shard_degrees(input), std::nullopt, relative_ff_dim_t{-1}))};
  FFOrdered<int> shard_degrees = FFOrdered<int>{get_discard_copy_degree(input)};

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
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1};
  FFOrdered<int> shard_degrees = ff_ordered_shard_degrees(input);
  shard_degrees.at(relative_ff_dim_t{-1}) = get_discard_copy_degree(input);

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

// tl::expected<ParallelTensorSpaceMapping, std::string>
//     get_input_to_projection_parallel_mapping(LinearAttrs const &attrs, 
//                                              ParallelTensorDimDegrees const &input) {
//   return ParallelTensor{
//
//   };
// }

tl::expected<ParallelTensorSpaceMapping, std::string>
    get_input_to_output_projection(LinearAttrs const &attrs, TensorNumDims const &input_num_dims) {
  
  auto inp_to_out = make_empty_down_projection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t>();

  project_dims(inp_to_out, 
               /*from=*/{sum_dim_idx(), shard_dim_idx(ff_dim_t{-1})},
               /*onto=*/sum_dim_idx());
  project_dims(inp_to_out, 
               /*from=*/{discard_copy_dim_idx()}, 
               /*onto=*/shard_dim_idx(ff_dim_t{-1}));

  for (ff_dim_t const &idx : ff_dim_range(input_num_dims.value - 1)) {
    project_dims(inp_to_out, 
                 /*from=*/{shard_dim_idx(idx)}, 
                 /*onto=*/shard_dim_idx(idx));
  }

  return ParallelTensorSpaceMapping{DimProjection{inp_to_out}};
}

tl::expected<OperatorSpaceParallelTensorSpaceMapping, std::string>
  get_operator_to_input_projection(LinearAttrs const &attrs,
                                   TensorNumDims const &input_num_dims) {

    TensorNumDims output_num_dims = input_num_dims;

    UpProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t> 
      out_to_inp = invert_down_projection(throw_if_unexpected(get_input_to_output_projection(attrs, input_num_dims)).raw_projection.require_down_proj());

    EqProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t> 
      op_to_out = throw_if_unexpected(get_output_space_mapping(attrs, output_num_dims)).raw_projection.require_eq_proj();

    return OperatorSpaceParallelTensorSpaceMapping{
      DimProjection{
        compose_up_projections(up_from_eq_proj(op_to_out), out_to_inp),
      },
    };
}

tl::expected<OperatorSpaceParallelTensorSpaceMapping, std::string>
    get_operator_to_output_projection(LinearAttrs const &attrs,
                                      TensorNumDims const &output_num_dims) {

  return OperatorSpaceParallelTensorSpaceMapping{
    get

  SumDegree sum_degree = SumDegree{1};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      get_sum_degree(input) *
      product(
          slice(ff_ordered_shard_degrees(input), std::nullopt, ff_dim_t{-1}))};
  FFOrdered<int> shard_degrees = FFOrdered<int>{
      shard_dim_at_idx(input, ff_dim_t{-1}).degree,
      get_discard_copy_degree(input),
  };

  return 
}

tl::expected<OperatorSpaceParallelTensorSpaceMapping, std::string>
    get_output_space_mapping(LinearAttrs const &attrs, 
                             ParallelTensorDimDegrees const &input) {
  return get_identity_mapping(input);
}

} // namespace FlexFlow
