#include "substitutions/operator_pattern/get_attribute.h"
#include "op-attrs/get_op_type.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

std::optional<OperatorAttributeValue> get_attribute(BatchMatmulAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(BatchNormAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::EPSILON:
      return OperatorAttributeValue{p.eps};
    case OperatorAttributeKey::AFFINE:
      return OperatorAttributeValue{p.affine};
    case OperatorAttributeKey::MOMENTUM:
      return OperatorAttributeValue{p.momentum};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(BroadcastAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::TARGET_DIMS:
      return OperatorAttributeValue{p.target_dims};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(CastAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::DATA_TYPE:
      return OperatorAttributeValue{p.dtype};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(CombineAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return OperatorAttributeValue{p.combine_dim};
    case OperatorAttributeKey::PARALLEL_DIM:
      return OperatorAttributeValue{p.combine_degree};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ConcatAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::AXIS:
      return OperatorAttributeValue{p.axis};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(Conv2DAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::KERNEL_H:
      return OperatorAttributeValue{p.kernel_h};
    case OperatorAttributeKey::KERNEL_W:
      return OperatorAttributeValue{p.kernel_w};
    case OperatorAttributeKey::STRIDE_H:
      return OperatorAttributeValue{p.stride_h};
    case OperatorAttributeKey::STRIDE_W:
      return OperatorAttributeValue{p.stride_w};
    case OperatorAttributeKey::PADDING_H:
      return OperatorAttributeValue{p.padding_h};
    case OperatorAttributeKey::PADDING_W:
      return OperatorAttributeValue{p.padding_w};
    case OperatorAttributeKey::GROUPS:
      return OperatorAttributeValue{p.groups};
    case OperatorAttributeKey::ACTIVATION:
      return OperatorAttributeValue{p.activation};
    case OperatorAttributeKey::USE_BIAS:
      return OperatorAttributeValue{p.use_bias};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ElementBinaryAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ElementUnaryAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(DropoutAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(EmbeddingAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::DATA_TYPE:
      return OperatorAttributeValue{p.data_type};
    case OperatorAttributeKey::AGGR:
      return OperatorAttributeValue{p.aggr};
    case OperatorAttributeKey::NUM_ENTRIES:
      return OperatorAttributeValue{p.num_entries};
    case OperatorAttributeKey::OUT_CHANNELS:
      return OperatorAttributeValue{p.out_channels};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(FlatAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(GatherAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::AXIS:
      return OperatorAttributeValue{p.dim};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(InputAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(LayerNormAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::AFFINE:
      return OperatorAttributeValue{p.elementwise_affine};
    case OperatorAttributeKey::AXES:
      return OperatorAttributeValue{vector_of(p.axes)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(LinearAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::OUT_CHANNELS:
      return OperatorAttributeValue{p.out_channels};
    case OperatorAttributeKey::USE_BIAS:
      return OperatorAttributeValue{p.use_bias};
    case OperatorAttributeKey::DATA_TYPE:
      return OperatorAttributeValue{p.data_type};
    case OperatorAttributeKey::ACTIVATION:
      return OperatorAttributeValue{p.activation};
    case OperatorAttributeKey::REGULARIZER:
      return OperatorAttributeValue{p.regularizer};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue>
    get_attribute(MultiHeadAttentionAttrs const &p, OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::NUM_HEADS:
      return OperatorAttributeValue{p.num_heads};
    case OperatorAttributeKey::USE_BIAS:
      return OperatorAttributeValue{p.bias};
    case OperatorAttributeKey::DROPOUT:
      return OperatorAttributeValue{p.dropout};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(NoopAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(Pool2DAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::KERNEL_H:
      return OperatorAttributeValue{p.kernel_h};
    case OperatorAttributeKey::KERNEL_W:
      return OperatorAttributeValue{p.kernel_w};
    case OperatorAttributeKey::STRIDE_H:
      return OperatorAttributeValue{p.stride_h};
    case OperatorAttributeKey::STRIDE_W:
      return OperatorAttributeValue{p.stride_w};
    case OperatorAttributeKey::PADDING_H:
      return OperatorAttributeValue{p.padding_h};
    case OperatorAttributeKey::PADDING_W:
      return OperatorAttributeValue{p.padding_w};
    case OperatorAttributeKey::POOL_TYPE:
      return OperatorAttributeValue{p.pool_type};
    case OperatorAttributeKey::ACTIVATION:
      return OperatorAttributeValue{p.activation};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ReduceAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ReductionAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return OperatorAttributeValue{p.reduction_degree};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(RepartitionAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::PARALLEL_OP_DIM:
      return OperatorAttributeValue{p.repartition_dim};
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return OperatorAttributeValue{p.repartition_degree};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ReplicateAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::PARALLEL_OP_DEGREE:
      return OperatorAttributeValue{p.replicate_degree};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ReshapeAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(ReverseAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::AXIS:
      return OperatorAttributeValue{p.axis};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(SplitAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::AXIS:
      return OperatorAttributeValue{p.axis};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(SoftmaxAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::AXIS:
      return OperatorAttributeValue{p.dim};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(TopKAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(TransposeAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    case OperatorAttributeKey::PERMUTATION:
      return OperatorAttributeValue{vector_of(p.perm)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(WeightAttrs const &p,
                                                    OperatorAttributeKey key) {
  switch (key) {
    case OperatorAttributeKey::OP_TYPE:
      return OperatorAttributeValue{get_op_type(p)};
    default:
      return std::nullopt;
  }
}

std::optional<OperatorAttributeValue> get_attribute(PCGOperatorAttrs const &p,
                                                    OperatorAttributeKey key) {
  return p.visit<std::optional<OperatorAttributeValue>>(
      [&](auto const &attrs) { return get_attribute(attrs, key); });
}

} // namespace FlexFlow
