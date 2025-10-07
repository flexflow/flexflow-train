#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_LOWER_OP_TASK_INVOCATION_TO_TASK_INVOCATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_LOWER_OP_TASK_INVOCATION_TO_TASK_INVOCATION_H

#include "task-spec/device_specific_per_device_op_state.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature.dtg.h"
#include "task-spec/symbolic_training_tensor_guid_t.dtg.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature_with_shapes.dtg.h"
#include "task-spec/fwb_tensor_slot_binding.dtg.h"
#include "task-spec/training_tensor_slot_binding.dtg.h"
#include "task-spec/symbolic_layer_tensor_shape_signature.dtg.h"

namespace FlexFlow {

TaskInvocation
  lower_op_task_invocation_to_task_invocation(
    OpTaskInvocation const &op_task_invocation,
    TrainingLayerSymbolicTensorGroupSignatureWithShapes const &layer_signature,
    std::optional<DeviceSpecificPerDeviceOpState> const &device_specific_device_states);

TrainingTensorSlotBinding
  lower_fwb_tensor_binding_to_training_tensor_binding(
    TrainingLayerSymbolicTensorGroupSignature const &training_layer_signature,
    FwbTensorSlotBinding const &fwb_slot_binding);

TaskArgSpec lower_op_arg_spec_to_task_arg_spec(
    OpArgSpec const &op_arg_spec,
    SymbolicLayerTensorShapeSignature const &op_shape_signature,
    std::optional<DeviceSpecificPerDeviceOpState> const
        &device_specific_device_states);

ConcreteArgSpec lower_runtime_arg_ref_spec_to_concrete_arg_spec(
    RuntimeArgRefSpec const &,
    RuntimeArgConfig const &);

ConcreteArgSpec lower_op_arg_ref_spec_to_concrete_arg_spec(
    OpArgRefSpec const &,
    SymbolicLayerTensorShapeSignature const &,
    std::optional<DeviceSpecificPerDeviceOpState> const &);

} // namespace FlexFlow

#endif
