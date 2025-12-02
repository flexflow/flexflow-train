#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_INVOCATION_RUNTIME_TASK_INVOCATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_INVOCATION_RUNTIME_TASK_INVOCATION_H

#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/ops/op_task_invocation.dtg.h"
#include "task-spec/ops/op_task_type.dtg.h"
#include "task-spec/runtime_task_invocation/runtime_task_invocation.dtg.h"
#include "task-spec/symbolic/symbolic_cg_op_attrs_and_training_signature_with_shapes.dtg.h"
#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature_with_shapes.dtg.h"
#include "task-spec/symbolic/symbolic_training_layer_attrs_plus_context.dtg.h"

namespace FlexFlow {

std::optional<RuntimeTaskInvocation>
  get_runtime_task_invocation_for_layer_and_type(symbolic_layer_guid_t,
                                                 SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &,
                                                 OpTaskType);

std::optional<RuntimeTaskInvocation>
  get_init_runtime_task_invocation_for_layer(symbolic_layer_guid_t, 
                                             SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &);

std::optional<RuntimeTaskInvocation>
  get_forward_runtime_task_invocation_for_layer(
      symbolic_layer_guid_t, 
      SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &);

std::optional<RuntimeTaskInvocation>
  get_backward_runtime_task_invocation_for_layer(symbolic_layer_guid_t,
                                                 SymbolicCgOpAttrsAndTrainingSignatureWithShapes const &);

RuntimeTaskInvocation
  get_compute_loss_runtime_task_invocation(LossAttrs const &,
                                           symbolic_forward_tensor_guid_t loss_fwd_tensor,
                                           symbolic_gradient_tensor_guid_t loss_grad_tensor,
                                           symbolic_loss_tensor_guid_t label_tensor);

std::optional<RuntimeTaskInvocation>
  get_update_runtime_task_invocation_for_layer(SymbolicTrainingLayerAttrsPlusContext const &,
                                               OptimizerAttrs const &);

RuntimeTaskInvocation
  lower_op_task_invocation_to_runtime_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    symbolic_layer_guid_t symbolic_layer_guid,
    SymbolicLayerTrainingTensorGroupSignatureWithShapes const &layer_signature);

}

#endif
