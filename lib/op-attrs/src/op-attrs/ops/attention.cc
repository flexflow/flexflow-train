#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.h"
#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/containers/extend.h"
#include "utils/integer_conversions.h"

namespace FlexFlow {

/* bool MultiHeadAttentionAttrs::is_valid(std::vector<ParallelTensorShape> const
 * &inputs) const { */
/*   return (inputs.size() == 3 && std::all_of(inputs.begin(), inputs.end(),
 * [](ParallelTensorShape const &s) { return s.is_valid(); })); */
/*   bool is_valid = true; */
/*   return is_valid; */
/* } */

nonnegative_int get_qProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.kdim;
}

nonnegative_int get_vProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.vdim;
}

nonnegative_int get_kProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.kdim;
}

nonnegative_int get_oProjSize(MultiHeadAttentionAttrs const &attrs) {
  return attrs.embed_dim;
}

nonnegative_int get_qSize(TensorShape const &query_shape) {
  return dim_at_idx(query_shape, relative_ff_dim_t{0});
}

nonnegative_int get_kSize(TensorShape const &key_shape) {
  return dim_at_idx(key_shape, relative_ff_dim_t{0});
}

nonnegative_int get_vSize(TensorShape const &value_shape) {
  return dim_at_idx(value_shape, relative_ff_dim_t{0});
}

nonnegative_int get_qSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.query_dim.size;
}

nonnegative_int get_qSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.query_size;
}

nonnegative_int get_kSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.key_dim.size;
}

nonnegative_int get_kSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.key_size;
}

nonnegative_int get_vSize(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.value_dim.size;
}

nonnegative_int get_vSize(MultiHeadAttentionInputs const &inputs) {
  return inputs.value_size;
}

nonnegative_int
    get_kvSeqLength(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.sequence_dim.size;
}

nonnegative_int get_kvSeqLength(MultiHeadAttentionInputs const &inputs) {
  return inputs.sequence_length;
}

nonnegative_int
    get_qoSeqLength(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.sequence_dim.size; // FIXME -- assumes only prefill
}

nonnegative_int get_qoSeqLength(MultiHeadAttentionInputs const &inputs) {
  return inputs.sequence_length; // FIXME -- assumes only prefil
}

nonnegative_int
    get_num_samples(MultiHeadAttentionParallelInputs const &inputs) {
  return inputs.batch_dim.size;
}

nonnegative_int get_num_samples(MultiHeadAttentionInputs const &inputs) {
  return inputs.batch_size;
}

std::vector<IncomingTensorRole>
    get_attention_incoming_tensor_roles(MultiHeadAttentionAttrs const &attrs) {

  std::vector<IncomingTensorRole> roles = std::vector{
      IncomingTensorRole::INPUT,
      IncomingTensorRole::INPUT,
      IncomingTensorRole::INPUT,
      IncomingTensorRole::WEIGHT,
  };

  if (attrs.bias) {
    extend(roles,
           std::vector{IncomingTensorRole::WEIGHT, IncomingTensorRole::WEIGHT});
  }

  return roles;
}

tl::expected<TensorShape, std::string>
    get_output_shape(MultiHeadAttentionAttrs const &attrs,
                     TensorShape const &input_q,
                     TensorShape const &input_k,
                     TensorShape const &input_v) {
  tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
      parse_attention_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }

  MultiHeadAttentionInputs parsed = parse_result.value();

  return TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{
          parsed.batch_size,
          parsed.sequence_length,
          attrs.embed_dim,
      }},
      parsed.datatype,
  };
}

tl::expected<TensorShape, std::string>
    get_weights_shape(MultiHeadAttentionAttrs const &attrs,
                      TensorShape const &input_q,
                      TensorShape const &input_k,
                      TensorShape const &input_v) {
  tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
      parse_attention_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }

  MultiHeadAttentionInputs parsed = parse_result.value();

  // W^Q_i in "Attention Is All You Need" top of page 5
  nonnegative_int qProjectWeightSize = parsed.query_size * attrs.kdim;

  // W^K_i in "Attention Is All You Need" top of page 5 (all i's put together)
  nonnegative_int kProjectWeightSize = parsed.key_size * attrs.kdim;

  // W^V_i in "Attention Is All You Need" top of page 5 (all i's put together)
  nonnegative_int vProjectWeightSize = parsed.value_size * attrs.vdim;

  // W^O in "Attention Is All You Need" top of page 5, with num_heads factored
  // out
  nonnegative_int outWeightSize = attrs.vdim * attrs.embed_dim;

  return TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{
          (qProjectWeightSize + kProjectWeightSize + vProjectWeightSize +
           outWeightSize),
          attrs.num_heads,
      }},
      parsed.datatype,
  };
}

tl::expected<TensorShape, std::string>
    get_input_bias_shape(MultiHeadAttentionAttrs const &attrs,
                         TensorShape const &input_q,
                         TensorShape const &input_k,
                         TensorShape const &input_v) {
  MultiHeadAttentionInputs parsed = ({
    tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
        parse_attention_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }
    parse_result.value();
  });

  return TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{
          attrs.kdim + attrs.kdim + attrs.vdim,
      }},
      parsed.datatype,
  };
}

tl::expected<TensorShape, std::string>
    get_output_bias_shape(MultiHeadAttentionAttrs const &attrs,
                          TensorShape const &input_q,
                          TensorShape const &input_k,
                          TensorShape const &input_v) {
  MultiHeadAttentionInputs parsed = ({
    tl::expected<MultiHeadAttentionInputs, std::string> parse_result =
        parse_attention_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }
    parse_result.value();
  });

  return TensorShape{
      TensorDims{FFOrdered<nonnegative_int>{
          attrs.embed_dim,
      }},
      parsed.datatype,
  };
}

tl::expected<ParallelTensorShape, std::string>
    get_weights_shape(MultiHeadAttentionAttrs const &attrs,
                      ParallelTensorShape const &input_q,
                      ParallelTensorShape const &input_k,
                      ParallelTensorShape const &input_v) {
  tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }
  MultiHeadAttentionParallelInputs parsed = parse_result.value();

  tl::expected<TensorShape, std::string> result_unpar_get_shape =
      get_weights_shape(attrs,
                        get_reduced_shape(input_q),
                        get_reduced_shape(input_k),
                        get_reduced_shape(input_v));
  if (!result_unpar_get_shape.has_value()) {
    return tl::unexpected(result_unpar_get_shape.error());
  }
  TensorShape unpar_shape = result_unpar_get_shape.value();

  nonnegative_int joined_dim_degree = 1_n;
  nonnegative_int head_dim_degree = parsed.discard_copy_degree.value;

  return lift_to_parallel_with_degrees(
      unpar_shape,
      SumDegree{1_n},
      DiscardCopyDegree{parsed.batch_dim.degree},
      FFOrdered<nonnegative_int>{joined_dim_degree, head_dim_degree});
}

tl::expected<ParallelTensorShape, std::string>
    get_input_bias_shape(MultiHeadAttentionAttrs const &attrs,
                         ParallelTensorShape const &input_q,
                         ParallelTensorShape const &input_k,
                         ParallelTensorShape const &input_v) {
  MultiHeadAttentionParallelInputs parsed = ({
    tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
        parse_attention_parallel_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }

    parse_result.value();
  });

  TensorShape unpar_shape = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_input_bias_shape(attrs,
                             get_reduced_shape(input_q),
                             get_reduced_shape(input_k),
                             get_reduced_shape(input_v));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }

    result_unpar.value();
  });

  SumDegree sum_degree = SumDegree{1_n};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      parsed.batch_dim.degree * parsed.discard_copy_degree.value};
  FFOrdered<nonnegative_int> shard_degrees = FFOrdered<nonnegative_int>{1_n};
  return lift_to_parallel_with_degrees(
      unpar_shape, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_output_bias_shape(MultiHeadAttentionAttrs const &attrs,
                          ParallelTensorShape const &input_q,
                          ParallelTensorShape const &input_k,
                          ParallelTensorShape const &input_v) {
  MultiHeadAttentionParallelInputs parsed = ({
    tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
        parse_attention_parallel_input_shape(input_q, input_k, input_v);
    if (!parse_result.has_value()) {
      return tl::unexpected(parse_result.error());
    }

    parse_result.value();
  });

  TensorShape unpar_shape = ({
    tl::expected<TensorShape, std::string> result_unpar =
        get_output_bias_shape(attrs,
                              get_reduced_shape(input_q),
                              get_reduced_shape(input_k),
                              get_reduced_shape(input_v));
    if (!result_unpar.has_value()) {
      return tl::unexpected(result_unpar.error());
    }

    result_unpar.value();
  });

  SumDegree sum_degree = SumDegree{1_n};
  DiscardCopyDegree discard_copy_degree = DiscardCopyDegree{
      parsed.batch_dim.degree * parsed.discard_copy_degree.value};
  FFOrdered<nonnegative_int> shard_degrees = FFOrdered<nonnegative_int>{1_n};
  return lift_to_parallel_with_degrees(
      unpar_shape, sum_degree, discard_copy_degree, shard_degrees);
}

tl::expected<ParallelTensorShape, std::string>
    get_output_shape(MultiHeadAttentionAttrs const &attrs,
                     ParallelTensorShape const &input_q,
                     ParallelTensorShape const &input_k,
                     ParallelTensorShape const &input_v) {
  tl::expected<MultiHeadAttentionParallelInputs, std::string> parse_result =
      parse_attention_parallel_input_shape(input_q, input_k, input_v);
  if (!parse_result.has_value()) {
    return tl::unexpected(parse_result.error());
  }
  MultiHeadAttentionParallelInputs parsed = parse_result.value();

  tl::expected<TensorShape, std::string> result_unpar_get_shape =
      get_output_shape(attrs,
                       get_reduced_shape(input_q),
                       get_reduced_shape(input_k),
                       get_reduced_shape(input_v));
  if (!result_unpar_get_shape.has_value()) {
    return tl::unexpected(result_unpar_get_shape.error());
  }
  TensorShape unpar_shape = result_unpar_get_shape.value();

  nonnegative_int sum_degree = parsed.discard_copy_degree.value;
  nonnegative_int discard_copy_degree = 1_n;
  nonnegative_int batch_degree = parsed.batch_dim.degree;
  nonnegative_int seq_len_degree = 1_n;
  nonnegative_int out_dim_degree = 1_n;

  return lift_to_parallel_with_degrees(
      unpar_shape,
      SumDegree{sum_degree},
      DiscardCopyDegree{discard_copy_degree},
      FFOrdered<nonnegative_int>{batch_degree, seq_len_degree, out_dim_degree});
}

nonnegative_int get_oSize(ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}

nonnegative_int get_oSize(TensorShape const &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
