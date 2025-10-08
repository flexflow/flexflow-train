#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_LOWER_OP_TASK_INVOCATION_TO_RUNTIME_TASK_INVOCATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_LOWER_OP_TASK_INVOCATION_TO_RUNTIME_TASK_INVOCATION_H

#include "task-spec/device_specific_per_device_op_state.dtg.h"
#include "task-spec/op_arg_ref_spec.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/runtime_task_invocation.dtg.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature.dtg.h"
#include "task-spec/symbolic_training_tensor_guid_t.dtg.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature_with_shapes.dtg.h"
#include "task-spec/fwb_tensor_slot_binding.dtg.h"
#include "task-spec/training_tensor_slot_binding.dtg.h"
#include "task-spec/symbolic_layer_tensor_shape_signature.dtg.h"

namespace FlexFlow {

RuntimeTaskInvocation
  lower_op_task_invocation_to_runtime_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    symbolic_layer_guid_t symbolic_layer_guid,
    TrainingLayerSymbolicTensorGroupSignatureWithShapes const &layer_signature);

TrainingTensorSlotBinding
  lower_fwb_tensor_binding_to_training_tensor_binding(
    TrainingLayerSymbolicTensorGroupSignature const &training_layer_signature,
    FwbTensorSlotBinding const &fwb_slot_binding);

RuntimeArgSpec lower_op_arg_spec_to_runtime_arg_spec(
    OpArgSpec const &op_arg_spec,
    symbolic_layer_guid_t symbolic_layer_guid,
    SymbolicLayerTensorShapeSignature const &op_shape_signature);

RuntimeArgSpec lower_op_arg_ref_spec_to_runtime_arg_spec(
    OpArgRefSpec const &,
    symbolic_layer_guid_t symbolic_layer_guid,
    SymbolicLayerTensorShapeSignature const &);

// TODO(@lockshaw)(#pr): this really shouldn't be here
ConcreteArgSpec lower_runtime_arg_ref_spec_to_concrete_arg_spec(
    RuntimeArgRefSpec const &,
    RuntimeArgConfig const &);

} // namespace FlexFlow

#endif
