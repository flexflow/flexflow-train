#include "task-spec/task_id_with_noop_default_t.h"
#include "utils/overload.h"

namespace FlexFlow {

task_id_with_noop_default_t default_noop_task() {
  return task_id_with_noop_default_t{std::monostate{}};
}

task_id_with_noop_default_t
  lower_op_task_id_to_task_id_with_noop_default_t(op_task_id_t op_task_id, 
                                                  ComputationGraphOpAttrs const &op_attrs) {
  switch (op_task_id) {
    case op_task_id_t::INIT:
      return get_init_task_id_for_op_attrs(op_attrs);
    case op_task_id_t::FWD:
      return get_fwd_task_id_for_op_attrs(op_attrs);
    case op_task_id_t::BWD:
      return get_bwd_task_id_for_op_attrs(op_attrs);
  }
}

task_id_with_noop_default_t
  get_init_task_id_for_op_attrs(ComputationGraphOpAttrs const &op_attrs) {
  
  return op_attrs.visit<task_id_with_noop_default_t>(overload {
    [](BatchMatmulAttrs const &) { return default_noop_task(); },
    [](BatchNormAttrs const &) { return lift_task_id_t(task_id_t::BATCHNORM_INIT_TASK_ID); },
    [](CastAttrs const &) { return default_noop_task(); },
    [](ConcatAttrs const &) { return default_noop_task(); },
    [](Conv2DAttrs const &) { return lift_task_id_t(task_id_t::CONV2D_INIT_TASK_ID); },
    [](DropoutAttrs const &) { return lift_task_id_t(task_id_t::DROPOUT_INIT_TASK_ID); },
    [](ElementBinaryAttrs const &) { return lift_task_id_t(task_id_t::ELEMENTBINARY_INIT_TASK_ID); },
    [](ElementUnaryAttrs const &) { return lift_task_id_t(task_id_t::ELEMENTBINARY_INIT_TASK_ID); },
    [](EmbeddingAttrs const &) { return default_noop_task(); },
    [](GatherAttrs const &) { return lift_task_id_t(task_id_t::GATHER_INIT_TASK_ID); },
    [](InputAttrs const &) { return default_noop_task(); },
    [](LayerNormAttrs const &) { return lift_task_id_t(task_id_t::LAYERNORM_INIT_TASK_ID); },
    [](LinearAttrs const &) { return lift_task_id_t(task_id_t::LINEAR_INIT_TASK_ID); },
    [](MultiHeadAttentionAttrs const &) { return lift_task_id_t(task_id_t::ATTENTION_INIT_TASK_ID); },
    [](Pool2DAttrs const &) { return lift_task_id_t(task_id_t::POOL2D_INIT_TASK_ID); },
    [](SoftmaxAttrs const &) { return lift_task_id_t(task_id_t::SOFTMAX_INIT_TASK_ID); },
    [](TransposeAttrs const &) { return default_noop_task(); },
    [](WeightAttrs const &) { return default_noop_task(); },
  });
}

task_id_with_noop_default_t
  get_fwd_task_id_for_op_attrs(ComputationGraphOpAttrs const &op_attrs) {
  
  return op_attrs.visit<task_id_with_noop_default_t>(overload {
    [](BatchMatmulAttrs const &) { return lift_task_id_t(task_id_t::BATCHMATMUL_FWD_TASK_ID); },
    [](BatchNormAttrs const &) { return lift_task_id_t(task_id_t::BATCHNORM_FWD_TASK_ID); },
    [](CastAttrs const &) { return lift_task_id_t(task_id_t::CAST_FWD_TASK_ID); },
    [](ConcatAttrs const &) { return lift_task_id_t(task_id_t::CONCAT_FWD_TASK_ID); },
    [](Conv2DAttrs const &) { return lift_task_id_t(task_id_t::CONV2D_FWD_TASK_ID); },
    [](DropoutAttrs const &) { return lift_task_id_t(task_id_t::DROPOUT_FWD_TASK_ID); },
    [](ElementBinaryAttrs const &) { return lift_task_id_t(task_id_t::ELEMENTBINARY_FWD_TASK_ID); },
    [](ElementUnaryAttrs const &) { return lift_task_id_t(task_id_t::ELEMENTBINARY_FWD_TASK_ID); },
    [](EmbeddingAttrs const &) { return lift_task_id_t(task_id_t::EMBED_FWD_TASK_ID); },
    [](FlatAttrs const &) { return lift_task_id_t(task_id_t::FLAT_FWD_TASK_ID); },
    [](GatherAttrs const &) { return lift_task_id_t(task_id_t::GATHER_FWD_TASK_ID); },
    [](InputAttrs const &) { return default_noop_task(); },
    [](LayerNormAttrs const &) { return lift_task_id_t(task_id_t::LAYERNORM_FWD_TASK_ID); },
    [](LinearAttrs const &) { return lift_task_id_t(task_id_t::LINEAR_FWD_TASK_ID); },
    [](MultiHeadAttentionAttrs const &) { return lift_task_id_t(task_id_t::ATTENTION_FWD_TASK_ID); },
    [](Pool2DAttrs const &) { return lift_task_id_t(task_id_t::POOL2D_FWD_TASK_ID); },
    [](SoftmaxAttrs const &) { return lift_task_id_t(task_id_t::SOFTMAX_FWD_TASK_ID); },
    [](TransposeAttrs const &) { return lift_task_id_t(task_id_t::TRANSPOSE_FWD_TASK_ID); },
    [](WeightAttrs const &) { return default_noop_task(); },
  });
}

task_id_with_noop_default_t
  get_bwd_task_id_for_op_attrs(ComputationGraphOpAttrs const &op_attrs) {
  
  return op_attrs.visit<task_id_with_noop_default_t>(overload {
    [](BatchMatmulAttrs const &) { return lift_task_id_t(task_id_t::BATCHMATMUL_BWD_TASK_ID); },
    [](BatchNormAttrs const &) { return lift_task_id_t(task_id_t::BATCHNORM_BWD_TASK_ID); },
    [](CastAttrs const &) { return lift_task_id_t(task_id_t::CAST_BWD_TASK_ID); },
    [](ConcatAttrs const &) { return lift_task_id_t(task_id_t::CONCAT_BWD_TASK_ID); },
    [](Conv2DAttrs const &) { return lift_task_id_t(task_id_t::CONV2D_BWD_TASK_ID); },
    [](DropoutAttrs const &) { return lift_task_id_t(task_id_t::DROPOUT_BWD_TASK_ID); },
    [](ElementBinaryAttrs const &) { return lift_task_id_t(task_id_t::ELEMENTBINARY_BWD_TASK_ID); },
    [](ElementUnaryAttrs const &) { return lift_task_id_t(task_id_t::ELEMENTBINARY_BWD_TASK_ID); },
    [](EmbeddingAttrs const &) { return lift_task_id_t(task_id_t::EMBED_BWD_TASK_ID); },
    [](FlatAttrs const &) { return lift_task_id_t(task_id_t::FLAT_BWD_TASK_ID); },
    [](GatherAttrs const &) { return lift_task_id_t(task_id_t::GATHER_BWD_TASK_ID); },
    [](InputAttrs const &) { return default_noop_task(); },
    [](LayerNormAttrs const &) { return lift_task_id_t(task_id_t::LAYERNORM_BWD_TASK_ID); },
    [](LinearAttrs const &) { return lift_task_id_t(task_id_t::LINEAR_BWD_TASK_ID); },
    [](MultiHeadAttentionAttrs const &) { return lift_task_id_t(task_id_t::ATTENTION_BWD_TASK_ID); },
    [](Pool2DAttrs const &) { return lift_task_id_t(task_id_t::POOL2D_BWD_TASK_ID); },
    [](SoftmaxAttrs const &) { return lift_task_id_t(task_id_t::SOFTMAX_BWD_TASK_ID); },
    [](TransposeAttrs const &) { return lift_task_id_t(task_id_t::TRANSPOSE_BWD_TASK_ID); },
    [](WeightAttrs const &) { return default_noop_task(); },
  });
}

} // namespace FlexFlow
