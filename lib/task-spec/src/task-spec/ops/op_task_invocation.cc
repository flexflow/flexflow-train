#include "task-spec/ops/op_task_invocation.h"
#include "task-spec/ops/impl/attention.h"
#include "task-spec/ops/impl/batch_matmul.h"
#include "task-spec/ops/impl/batch_norm.h"
#include "task-spec/ops/impl/cast.h"
#include "task-spec/ops/impl/concat.h"
#include "task-spec/ops/impl/conv_2d.h"
#include "task-spec/ops/impl/dropout.h"
#include "task-spec/ops/impl/element_binary.h"
#include "task-spec/ops/impl/element_unary.h"
#include "task-spec/ops/impl/embedding.h"
#include "task-spec/ops/impl/flat.h"
#include "task-spec/ops/impl/gather.h"
#include "task-spec/ops/impl/input.h"
#include "task-spec/ops/impl/layer_norm.h"
#include "task-spec/ops/impl/linear.h"
#include "task-spec/ops/impl/noop.h"
#include "task-spec/ops/impl/pool_2d.h"
#include "task-spec/ops/impl/reduce.h"
#include "task-spec/ops/impl/reshape.h"
#include "task-spec/ops/impl/reverse.h"
#include "task-spec/ops/impl/softmax.h"
#include "task-spec/ops/impl/split.h"
#include "task-spec/ops/impl/topk.h"
#include "task-spec/ops/impl/transpose.h"
#include "task-spec/ops/impl/weight.h"
#include "utils/overload.h"

namespace FlexFlow {

std::optional<OpTaskInvocation>
    get_init_op_task_invocation(ComputationGraphOpAttrs const &op) {
  return op.visit<OpTaskInvocation>(overload{
      [](BatchNormAttrs const &attrs) { return init(attrs); },
      [](Conv2DAttrs const &attrs) { return init(attrs); },
      [](DropoutAttrs const &attrs) { return init(attrs); },
      [](ElementBinaryAttrs const &attrs) { return init(attrs); },
      [](ElementUnaryAttrs const &attrs) { return init(attrs); },
      [](GatherAttrs const &attrs) { return init(attrs); },
      [](LayerNormAttrs const &attrs) { return init(attrs); },
      [](LinearAttrs const &attrs) { return init(attrs); },
      [](MultiHeadAttentionAttrs const &attrs) { return init(attrs); },
      [](Pool2DAttrs const &attrs) { return init(attrs); },
      [](ReduceAttrs const &attrs) { return init(attrs); },
      [](SoftmaxAttrs const &attrs) { return init(attrs); },
      [](auto const &attrs) -> OpTaskInvocation {
        PANIC("Unhandled attr type", attrs);
      },
  });
}

std::optional<OpTaskInvocation>
    get_forward_op_task_invocation(ComputationGraphOpAttrs const &op) {
  return op.visit<OpTaskInvocation>(overload{
      [](BatchMatmulAttrs const &attrs) { return forward(attrs); },
      [](BatchNormAttrs const &attrs) { return forward(attrs); },
      [](CastAttrs const &attrs) { return forward(attrs); },
      [](ConcatAttrs const &attrs) { return forward(attrs); },
      [](Conv2DAttrs const &attrs) { return forward(attrs); },
      [](DropoutAttrs const &attrs) { return forward(attrs); },
      [](ElementBinaryAttrs const &attrs) { return forward(attrs); },
      [](ElementUnaryAttrs const &attrs) { return forward(attrs); },
      [](EmbeddingAttrs const &attrs) { return forward(attrs); },
      [](FlatAttrs const &attrs) { return forward(attrs); },
      [](GatherAttrs const &attrs) { return forward(attrs); },
      [](LayerNormAttrs const &attrs) { return forward(attrs); },
      [](LinearAttrs const &attrs) { return forward(attrs); },
      [](MultiHeadAttentionAttrs const &attrs) { return forward(attrs); },
      [](Pool2DAttrs const &attrs) { return forward(attrs); },
      [](ReduceAttrs const &attrs) { return forward(attrs); },
      [](ReverseAttrs const &attrs) { return forward(attrs); },
      [](ReshapeAttrs const &attrs) { return forward(attrs); },
      [](SplitAttrs const &attrs) { return forward(attrs); },
      [](SoftmaxAttrs const &attrs) { return forward(attrs); },
      [](TopKAttrs const &attrs) { return forward(attrs); },
      [](TransposeAttrs const &attrs) { return forward(attrs); },
      [](auto const &attrs) -> OpTaskInvocation {
        PANIC("Unhandled attr type", attrs);
      },
  });
}

std::optional<OpTaskInvocation>
    get_backward_op_task_invocation(ComputationGraphOpAttrs const &op) {
  return op.visit<OpTaskInvocation>(overload{
      [](BatchMatmulAttrs const &attrs) { return backward(attrs); },
      [](BatchNormAttrs const &attrs) { return backward(attrs); },
      [](CastAttrs const &attrs) { return backward(attrs); },
      [](ConcatAttrs const &attrs) { return backward(attrs); },
      [](Conv2DAttrs const &attrs) { return backward(attrs); },
      [](DropoutAttrs const &attrs) { return backward(attrs); },
      [](ElementBinaryAttrs const &attrs) { return backward(attrs); },
      [](ElementUnaryAttrs const &attrs) { return backward(attrs); },
      [](EmbeddingAttrs const &attrs) { return backward(attrs); },
      [](FlatAttrs const &attrs) { return backward(attrs); },
      [](GatherAttrs const &attrs) { return backward(attrs); },
      [](LayerNormAttrs const &attrs) { return backward(attrs); },
      [](LinearAttrs const &attrs) { return backward(attrs); },
      [](MultiHeadAttentionAttrs const &attrs) { return backward(attrs); },
      [](Pool2DAttrs const &attrs) { return backward(attrs); },
      [](ReduceAttrs const &attrs) { return backward(attrs); },
      [](ReverseAttrs const &attrs) { return backward(attrs); },
      [](ReshapeAttrs const &attrs) { return backward(attrs); },
      [](SplitAttrs const &attrs) { return backward(attrs); },
      [](SoftmaxAttrs const &attrs) { return backward(attrs); },
      [](TopKAttrs const &attrs) { return backward(attrs); },
      [](TransposeAttrs const &attrs) { return backward(attrs); },
      [](auto const &attrs) -> OpTaskInvocation {
        PANIC("Unhandled attr type", attrs);
      },
  });
}

std::optional<OpTaskInvocation> get_op_task_invocation(
   ComputationGraphOpAttrs const &op_attrs, 
   OpTaskType task_type) {
  switch (task_type) {
    case OpTaskType::INIT:
      return get_init_op_task_invocation(op_attrs);
    case OpTaskType::FWD:
      return get_forward_op_task_invocation(op_attrs);
    case OpTaskType::BWD:
      return get_backward_op_task_invocation(op_attrs);
    default:
      PANIC("Unhandled OpTaskType", op_attrs);
  };
}

bool is_tensor_invocation_valid(OpTaskSignature const &sig,
                                OpTaskInvocation const &inv) {
  // TODO: fix for variadic inputs (need to implement .bind() for variadic
  // first)
  for (std::pair<fwb_tensor_slot_id_t, OpTensorSpec> const &tensor_binding :
       inv.binding.get_tensor_bindings()) {
    OpTensorSlotSpec op_tensor_slot_spec =
        OpTensorSlotSpec{tensor_binding.first.slot_id,
                         TensorSlotArity::TENSOR,
                         tensor_binding.second.role,
                         tensor_binding.first.is_grad,
                         tensor_binding.second.slot_option};

    if (!sig.get_tensor_slots().count(op_tensor_slot_spec)) {
      return false;
    }
  }

  return true;
}

bool is_arg_invocation_valid(OpTaskSignature const &sig,
                             OpTaskInvocation const &inv) {
  // TODO: fix for device specific args
  // for (std::pair<slot_id_t, OpArgSpec> const & arg_binding :
  // inv.binding.get_arg_bindings()) {
  //   if (sig.get_arg_types().count(arg_binding.first)) {
  //     if (get_op_arg_spec_type_index(arg_binding.second) !=
  //     sig.get_arg_types().at(arg_binding.first)) {
  //       return false;
  //     }
  //   } else {
  //     return false;
  //   }
  // }

  return true;
}

bool is_invocation_valid(OpTaskSignature const &sig,
                         OpTaskInvocation const &inv) {
  return is_tensor_invocation_valid(sig, inv) &&
         is_arg_invocation_valid(sig, inv);
}


} // namespace FlexFlow
