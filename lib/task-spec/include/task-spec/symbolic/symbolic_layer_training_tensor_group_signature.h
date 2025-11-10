#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_SYMBOLIC_LAYER_TRAINING_TENSOR_GROUP_SIGNATURE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_SYMBOLIC_LAYER_TRAINING_TENSOR_GROUP_SIGNATURE_H

#include "op-attrs/tensor_role.dtg.h"
#include "task-spec/fwb_tensor_type.dtg.h"
#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature.dtg.h"
#include "task-spec/symbolic/symbolic_training_tensor_guid_t.dtg.h"
#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

std::vector<SymbolicTrainingTensorGroup> get_training_tensor_groups_for_role(
    SymbolicLayerTrainingTensorGroupSignature const &signature, TensorRole tensor_role);

SymbolicTrainingTensorGroup get_training_tensor_group_for_role_and_index(
    SymbolicLayerTrainingTensorGroupSignature const &signature,
    TensorRole tensor_role,
    nonnegative_int index);

std::vector<symbolic_training_tensor_guid_t>
  get_training_tensors_for_role_and_type(SymbolicLayerTrainingTensorGroupSignature const &,
                                         TensorRole,
                                         FwbTensorType);


} // namespace FlexFlow

#endif
