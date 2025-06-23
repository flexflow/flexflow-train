#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_LAYER_TENSOR_GROUP_SIGNATURE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_LAYER_TENSOR_GROUP_SIGNATURE_H

#include "pcg/tensor_role.dtg.h"
#include "task-spec/training_layer_tensor_group_signature.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

std::vector<TrainingTensorGroup> get_training_tensor_groups_for_role(
    TrainingLayerTensorGroupSignature const &signature, TensorRole tensor_role);

TrainingTensorGroup get_training_tensor_group_for_role_and_index(
    TrainingLayerTensorGroupSignature const &signature,
    TensorRole tensor_role,
    nonnegative_int index);

} // namespace FlexFlow

#endif
