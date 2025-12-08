#include "op-attrs/get_incoming_tensor_roles.h"
#include "op-attrs/ops/attention.h"
#include "op-attrs/ops/batch_norm.h"
#include "op-attrs/ops/conv_2d.h"
#include "op-attrs/ops/layer_norm.h"
#include "op-attrs/ops/linear.h"
#include "op-attrs/pcg_operator_attrs.h"
#include "utils/overload.h"

namespace FlexFlow {

std::unordered_map<TensorSlotName, IncomingTensorRole> get_incoming_tensor_roles(
    ComputationGraphOpAttrs const &comp_graph_op_attrs) {
  return get_incoming_tensor_roles(
      pcg_op_attrs_from_compgraph_op_attrs(comp_graph_op_attrs));
}

std::unordered_map<TensorSlotName, IncomingTensorRole>
    get_incoming_tensor_roles(PCGOperatorAttrs const &pcg_op_attrs) {
  return pcg_op_attrs.visit<std::unordered_map<TensorSlotName, IncomingTensorRole>>(overload{
      [](BatchMatmulAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::LHS_INPUT, IncomingTensorRole::INPUT},
          {TensorSlotName::RHS_INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](BatchNormAttrs const &attrs) {
        return get_batch_norm_incoming_tensor_roles(attrs);
      },
      [](BroadcastAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](CastAttrs const &) { 
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        }; 
      },
      [](CombineAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [&](ConcatAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](Conv2DAttrs const &attrs) {
        return get_conv2d_incoming_tensor_roles(attrs);
      },
      [](DropoutAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](ElementBinaryAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::LHS_INPUT, IncomingTensorRole::INPUT},
          {TensorSlotName::RHS_INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](ElementUnaryAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](EmbeddingAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
          {TensorSlotName::WEIGHT, IncomingTensorRole::WEIGHT},
        };
      },
      [](FlatAttrs const &) { 
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](GatherAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](InputAttrs const &) { 
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{}; 
      },
      [](LayerNormAttrs const &attrs) {
        return get_layer_norm_incoming_tensor_roles(attrs);
      },
      [](LinearAttrs const &attrs) {
        return get_linear_incoming_tensor_roles(attrs);
      },
      [](MultiHeadAttentionAttrs const &attrs) {
        return get_attention_incoming_tensor_roles(attrs);
      },
      [](NoopAttrs const &) { 
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        }; 
      },
      [](Pool2DAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](ReduceAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](ReductionAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](RepartitionAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](ReplicateAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](ReverseAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](ReshapeAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](SplitAttrs const &) { 
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        }; 
      },
      [](SoftmaxAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](TopKAttrs const &) { 
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        }; 
      },
      [](TransposeAttrs const &) {
        return std::unordered_map<TensorSlotName, IncomingTensorRole>{
          {TensorSlotName::INPUT, IncomingTensorRole::INPUT},
        };
      },
      [](WeightAttrs const &) { return std::unordered_map<TensorSlotName, IncomingTensorRole>{}; },
  });
}

} // namespace FlexFlow
