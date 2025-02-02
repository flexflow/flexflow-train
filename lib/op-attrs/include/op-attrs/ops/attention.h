#ifndef _FLEXFLOW_ATTENTION_ATTRS_H
#define _FLEXFLOW_ATTENTION_ATTRS_H

#include "op-attrs/incoming_tensor_role.dtg.h"
#include "op-attrs/ops/attention/multihead_attention_inputs.dtg.h"
#include "op-attrs/ops/attention/multihead_attention_parallel_inputs.dtg.h"
#include "op-attrs/ops/attention_attrs.dtg.h"
#include "op-attrs/ops/core.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "op-attrs/tensor_shape.dtg.h"
#include <tl/expected.hpp>

namespace FlexFlow {

nonnegative_int get_qProjSize(MultiHeadAttentionAttrs const &);
nonnegative_int get_vProjSize(MultiHeadAttentionAttrs const &);
nonnegative_int get_kProjSize(MultiHeadAttentionAttrs const &);
nonnegative_int get_oProjSize(MultiHeadAttentionAttrs const &);

nonnegative_int get_qSize(MultiHeadAttentionParallelInputs const &);
nonnegative_int get_qSize(MultiHeadAttentionInputs const &);

nonnegative_int get_kSize(MultiHeadAttentionParallelInputs const &);
nonnegative_int get_kSize(MultiHeadAttentionInputs const &);

nonnegative_int get_vSize(MultiHeadAttentionParallelInputs const &);
nonnegative_int get_vSize(MultiHeadAttentionInputs const &);

nonnegative_int get_oSize(ParallelTensorShape const &);
nonnegative_int get_oSize(TensorShape const &);

nonnegative_int get_qoSeqLength(MultiHeadAttentionParallelInputs const &);
nonnegative_int get_qoSeqLength(MultiHeadAttentionInputs const &);

nonnegative_int get_kvSeqLength(MultiHeadAttentionParallelInputs const &);
nonnegative_int get_kvSeqLength(MultiHeadAttentionInputs const &);

nonnegative_int get_num_samples(MultiHeadAttentionParallelInputs const &);
nonnegative_int get_num_samples(MultiHeadAttentionInputs const &);

std::vector<IncomingTensorRole>
    get_attention_incoming_tensor_roles(MultiHeadAttentionAttrs const &);

tl::expected<TensorShape, std::string>
    get_weights_shape(MultiHeadAttentionAttrs const &,
                      TensorShape const &input_q,
                      TensorShape const &input_k,
                      TensorShape const &input_v);
tl::expected<TensorShape, std::string>
    get_input_bias_shape(MultiHeadAttentionAttrs const &,
                         TensorShape const &input_q,
                         TensorShape const &input_k,
                         TensorShape const &input_v);
tl::expected<TensorShape, std::string>
    get_output_bias_shape(MultiHeadAttentionAttrs const &,
                          TensorShape const &input_q,
                          TensorShape const &input_k,
                          TensorShape const &input_v);
tl::expected<TensorShape, std::string>
    get_output_shape(MultiHeadAttentionAttrs const &,
                     TensorShape const &input_q,
                     TensorShape const &input_k,
                     TensorShape const &input_v);

tl::expected<ParallelTensorDims, std::string>
    get_weights_parallel_dims(MultiHeadAttentionAttrs const &,
                              ParallelTensorShape const &input_q,
                              ParallelTensorShape const &input_k,
                              ParallelTensorShape const &input_v);
tl::expected<ParallelTensorDims, std::string>
    get_input_bias_parallel_dims(MultiHeadAttentionAttrs const &,
                                 ParallelTensorShape const &input_q,
                                 ParallelTensorShape const &input_k,
                                 ParallelTensorShape const &input_v);
tl::expected<ParallelTensorDims, std::string>
    get_output_bias_parallel_dims(MultiHeadAttentionAttrs const &,
                                  ParallelTensorShape const &input_q,
                                  ParallelTensorShape const &input_k,
                                  ParallelTensorShape const &input_v);

tl::expected<ParallelTensorShape, std::string>
    get_weights_shape(MultiHeadAttentionAttrs const &,
                      ParallelTensorShape const &input_q,
                      ParallelTensorShape const &input_k,
                      ParallelTensorShape const &input_v);
tl::expected<ParallelTensorShape, std::string>
    get_input_bias_shape(MultiHeadAttentionAttrs const &,
                         ParallelTensorShape const &input_q,
                         ParallelTensorShape const &input_k,
                         ParallelTensorShape const &input_v);
tl::expected<ParallelTensorShape, std::string>
    get_output_bias_shape(MultiHeadAttentionAttrs const &,
                          ParallelTensorShape const &input_q,
                          ParallelTensorShape const &input_k,
                          ParallelTensorShape const &input_v);
tl::expected<ParallelTensorShape, std::string>
    get_output_shape(MultiHeadAttentionAttrs const &,
                     ParallelTensorShape const &input_q,
                     ParallelTensorShape const &input_k,
                     ParallelTensorShape const &input_v);

CHECK_VALID_OP_ATTR(MultiHeadAttentionAttrs);
} // namespace FlexFlow

#endif
