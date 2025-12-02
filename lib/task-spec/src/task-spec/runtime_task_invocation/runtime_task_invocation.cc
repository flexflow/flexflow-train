#include "task-spec/runtime_task_invocation/runtime_task_invocation.h"
#include "task-spec/loss_functions.h"
#include "task-spec/optimizer.h"
#include "task-spec/runtime_task_invocation/runtime_arg_spec.h"
#include "task-spec/symbolic/symbolic_cg_op_attrs_and_training_signature_with_shapes.h"
#include "utils/containers/get_only.h"
#include "utils/containers/keys.h"
#include "task-spec/ops/op_task_invocation.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::optional<RuntimeTaskInvocation>
  get_init_runtime_task_invocation_for_layer(symbolic_layer_guid_t l,
                                             SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &attrs_and_signature) {

  ComputationGraphOpAttrs cg_op_attrs = attrs_and_signature.op_attrs;

  SymbolicLayerTrainingTensorGroupSignatureWithShapes layer_signature
    = get_signature_with_shapes(attrs_and_signature);

  OpTaskInvocation op_task_invocation = ({
    std::optional<OpTaskInvocation> maybe_invocation = get_init_op_task_invocation(cg_op_attrs);
    if (!maybe_invocation.has_value()) {
      return std::nullopt;
    }
    maybe_invocation.value();
  });

  return lower_op_task_invocation_to_runtime_task_invocation(
    /*op_task_invocation=*/op_task_invocation,
    /*symbolic_layer_guid=*/l,
    /*layer_signature=*/layer_signature);
}

std::optional<RuntimeTaskInvocation>
  get_forward_runtime_task_invocation_for_layer(
      symbolic_layer_guid_t layer_guid, 
      SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &attrs_and_signature) {

  ComputationGraphOpAttrs cg_op_attrs = attrs_and_signature.op_attrs;

  SymbolicLayerTrainingTensorGroupSignatureWithShapes layer_signature
    = get_signature_with_shapes(attrs_and_signature);

  OpTaskInvocation op_task_invocation = ({
    std::optional<OpTaskInvocation> maybe_invocation = get_forward_op_task_invocation(cg_op_attrs);
    if (!maybe_invocation.has_value()) {
      return std::nullopt;
    }
    maybe_invocation.value();
  });

  return lower_op_task_invocation_to_runtime_task_invocation(
    /*op_task_invocation=*/op_task_invocation,
    /*symbolic_layer_guid=*/layer_guid,
    /*layer_signature=*/layer_signature);
}

std::optional<RuntimeTaskInvocation>
  get_backward_task_invocation_for_layer(symbolic_layer_guid_t l,
                                         SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &attrs_and_signature) {

  ComputationGraphOpAttrs cg_op_attrs = attrs_and_signature.op_attrs;

  SymbolicLayerTrainingTensorGroupSignatureWithShapes layer_signature
    = get_signature_with_shapes(attrs_and_signature);


  OpTaskInvocation op_task_invocation = ({
    std::optional<OpTaskInvocation> maybe_invocation = get_backward_op_task_invocation(cg_op_attrs);
    if (!maybe_invocation.has_value()) {
      return std::nullopt;
    }
    maybe_invocation.value();
  });

  return lower_op_task_invocation_to_runtime_task_invocation(
    /*op_task_invocation=*/op_task_invocation,
    /*symbolic_layer_guid=*/l,
    /*layer_signature=*/layer_signature);
}

RuntimeTaskInvocation
  get_compute_loss_runtime_task_invocation(LossAttrs const &loss_attrs,
                                           symbolic_forward_tensor_guid_t loss_fwd_tensor,
                                           symbolic_gradient_tensor_guid_t loss_grad_tensor,
                                           symbolic_loss_tensor_guid_t label_tensor) {

  RuntimeTaskInvocation loss_invocation = loss_attrs_backward(
      loss_attrs,
      loss_fwd_tensor,
      loss_grad_tensor,
      label_tensor);

  return loss_invocation;
} 

std::optional<RuntimeTaskInvocation>
  get_update_runtime_task_invocation_for_layer(SymbolicTrainingLayerAttrsPlusContext const &training_layer,
                                               OptimizerAttrs const &optimizer_attrs) {
  if (training_layer.layer_attrs.op_attrs.has<WeightAttrs>()) {
    SymbolicTrainingTensorGroup weight_tensor_group =
        get_only(training_layer.output_tensor_groups);

    RuntimeTaskInvocation invocation =
        optimizer_attrs_get_update_invocation(
          optimizer_attrs,
          /*weight=*/weight_tensor_group.forward_tensor,
          /*weight_grad=*/weight_tensor_group.gradient_tensor,
          /*grad_buffer_tensors=*/weight_tensor_group.optimizer_tensors);

    return invocation;
  } else {
    return std::nullopt;
  }
}



RuntimeTaskInvocation
  lower_op_task_invocation_to_runtime_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    symbolic_layer_guid_t symbolic_layer_guid,
    SymbolicLayerTrainingTensorGroupSignatureWithShapes const &layer_signature) {

  // TODO(@lockshaw)(#pr): 
  NOT_IMPLEMENTED();
  // std::unordered_map<training_tensor_slot_id_t, symbolic_training_tensor_guid_t>
  //     tensor_bindings =
  //         transform(op_task_invocation.binding.get_tensor_bindings(),
  //                   [&](fwb_tensor_slot_id_t const &fwb_slot_id,
  //                       OpTensorSpec const &op_tensor_spec) {
  //                     FwbTensorSlotBinding fwb_binding = FwbTensorSlotBinding{
  //                       fwb_slot_id,
  //                       op_tensor_spec,
  //                     };
  //
  //                     TrainingTensorSlotBinding training_binding = 
  //                       lower_fwb_tensor_binding_to_training_tensor_binding(
  //                         drop_shapes_from_signature(layer_signature),
  //                         fwb_binding);
  //
  //                     return std::pair{
  //                       training_binding.slot,
  //                       training_binding.bound,
  //                     };
  //                   });
  //
  // std::unordered_map<slot_id_t, RuntimeArgSpec> arg_bindings = map_values(
  //     op_task_invocation.binding.get_arg_bindings(),
  //     [&](OpArgSpec const &op_arg_spec) -> RuntimeArgSpec {
  //       return lower_op_arg_spec_to_runtime_arg_spec(op_arg_spec,
  //                                                    symbolic_layer_guid,
  //                                                    get_shape_signature(layer_signature));
  //     });
  //
  // return RuntimeTaskInvocation{
  //     op_task_invocation.task_id,
  //     RuntimeTaskBinding{
  //       tensor_bindings,
  //       arg_bindings,
  //     },
  // };
}

} // namespace FlexFlow
