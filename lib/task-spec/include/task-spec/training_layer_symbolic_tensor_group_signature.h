#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_LAYER_SYMBOLIC_TENSOR_GROUP_SIGNATURE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_LAYER_SYMBOLIC_TENSOR_GROUP_SIGNATURE_H

#include "op-attrs/tensor_role.dtg.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

std::vector<SymbolicTrainingTensorGroup> get_training_tensor_groups_for_role(
    TrainingLayerSymbolicTensorGroupSignature const &signature, TensorRole tensor_role);

SymbolicTrainingTensorGroup get_training_tensor_group_for_role_and_index(
    TrainingLayerSymbolicTensorGroupSignature const &signature,
    TensorRole tensor_role,
    nonnegative_int index);

} // namespace FlexFlow

#endif
