#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/conv_2d/conv_2d_input_shape.h"
#include "op-attrs/ops/conv_2d/conv_2d_parallel_input_shape.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

std::vector<IncomingTensorRole>
    get_conv2d_incoming_tensor_roles(Conv2DAttrs const &attrs) {
  std::vector<IncomingTensorRole> result = {
      IncomingTensorRole::INPUT,
      IncomingTensorRole::WEIGHT,
  };

  if (attrs.use_bias) {
    result.push_back(IncomingTensorRole::WEIGHT);
  }

  return result;
}

TensorShape get_kernel_shape(Conv2DAttrs const &attrs,
                             TensorShape const &raw_input_shape) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported
  Conv2DInputShape input = parse_input_shape(raw_input_shape);

  return TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{
          attrs.out_channels,
          input.num_channels,
          attrs.kernel_h,
          attrs.kernel_w,
      }},
      input.datatype,
  };
}

TensorShape get_bias_shape(Conv2DAttrs const &attrs,
                           TensorShape const &raw_input_shape) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported
  Conv2DInputShape input = parse_input_shape(raw_input_shape);

  return TensorShape{
      TensorDims{
          FFOrdered<nonnegative_int>{attrs.out_channels},
      },
      input.datatype,
  };
}

static nonnegative_int calculate_output_size(nonnegative_int input_size,
                                             nonnegative_int padding_size,
                                             nonnegative_int kernel_size,
                                             nonnegative_int stride) {
  int input_size_raw = input_size.unwrap_nonnegative();
  int padding_raw = padding_size.unwrap_nonnegative();
  int kernel_size_raw = kernel_size.unwrap_nonnegative();
  int stride_raw = stride.unwrap_nonnegative();

  return nonnegative_int{
      (input_size_raw + (2 * padding_raw) - kernel_size_raw) / stride_raw + 1};
}

TensorShape get_output_shape(Conv2DAttrs const &attrs,
                             TensorShape const &raw_input_shape) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported
  Conv2DInputShape input = parse_input_shape(raw_input_shape);

  nonnegative_int out_height =
      calculate_output_size(/*input_size=*/input.height,
                            /*padding_size=*/attrs.padding_h,
                            /*kernel_size=*/attrs.kernel_h,
                            /*stride_size=*/attrs.stride_h);
  nonnegative_int out_width =
      calculate_output_size(/*input_size=*/input.width,
                            /*padding_size=*/attrs.padding_w,
                            /*kernel_size=*/attrs.kernel_w,
                            /*stride_size=*/attrs.stride_w);

  return TensorShape{TensorDims{FFOrdered<nonnegative_int>{
                         input.num_samples,
                         attrs.out_channels,
                         out_height,
                         out_width,
                     }},
                     input.datatype};
}

ParallelTensorShape get_kernel_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported

  Conv2DParallelInputShape parsed = parse_parallel_input_shape(input);

  TensorShape unpar = get_kernel_shape(attrs, get_reduced_shape(input));

  assert(parsed.height_dim.degree == 1);
  assert(parsed.width_dim.degree == 1);

  SumDegree sum_degree = SumDegree{1_n};
  DiscardCopyDegree discard_copy_degree =
      DiscardCopyDegree{parsed.sample_dim.degree * parsed.sum_reduction_degree};
  FFOrdered<nonnegative_int> shard_degrees = {
      parsed.discard_copy_reduction_degree,
      parsed.channel_dim.degree,
      1_n,
      1_n,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

ParallelTensorShape get_bias_shape(Conv2DAttrs const &attrs,
                                   ParallelTensorShape const &input) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported

  Conv2DParallelInputShape parsed = parse_parallel_input_shape(input);

  TensorShape unpar = get_bias_shape(attrs, get_reduced_shape(input));

  SumDegree sum_degree =
      SumDegree{parsed.sum_reduction_degree * parsed.channel_dim.degree};
  DiscardCopyDegree discard_copy_degree =
      DiscardCopyDegree{parsed.height_dim.degree * parsed.width_dim.degree *
                        parsed.sample_dim.degree};
  FFOrdered<nonnegative_int> shard_degrees = {
      parsed.discard_copy_reduction_degree,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

ParallelTensorShape get_output_shape(Conv2DAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  assert(attrs.groups == 1); // TODO(@lockshaw): currently not supported

  Conv2DParallelInputShape parsed = parse_parallel_input_shape(input);

  TensorShape unpar = get_output_shape(attrs, get_reduced_shape(input));

  assert(parsed.height_dim.degree == 1);
  assert(parsed.width_dim.degree == 1);

  SumDegree sum_degree =
      SumDegree{parsed.sum_reduction_degree * parsed.channel_dim.degree};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{1_n};
  FFOrdered<nonnegative_int> shard_degrees = {
      parsed.sample_dim.degree,
      parsed.discard_copy_reduction_degree,
      1_n,
      1_n,
  };

  return lift_to_parallel_with_degrees(
      unpar, sum_degree, discard_copy_degree, shard_degrees);
}

} // namespace FlexFlow
