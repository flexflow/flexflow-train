#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_TENSOR_SLOT_BINDING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_TENSOR_SLOT_BINDING_H

#include "task-spec/fwb_tensor_slot_binding.dtg.h"
#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature.dtg.h"
#include "task-spec/training_tensor_slot_binding.dtg.h"

namespace FlexFlow {

TrainingTensorSlotBinding
  lower_fwb_tensor_binding_to_training_tensor_binding(
    SymbolicLayerTrainingTensorGroupSignature const &training_layer_signature,
    FwbTensorSlotBinding const &fwb_slot_binding);

} // namespace FlexFlow

#endif
