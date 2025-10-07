#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_TRAINING_LAYER_PLUS_CONTEXT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_SYMBOLIC_TRAINING_LAYER_PLUS_CONTEXT_H

#include "op-attrs/tensor_role.dtg.h"
#include "task-spec/symbolic_training_layer_plus_context.dtg.h"
#include "task-spec/symbolic_training_tensor_group.dtg.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature.dtg.h"

namespace FlexFlow {

std::vector<SymbolicTrainingTensorGroup>
    get_training_tensor_groups_for_role(
        SymbolicTrainingLayerPlusContext const &training_layer_plus_context,
        TensorRole tensor_role);

SymbolicTrainingTensorGroup
    get_training_tensor_group_for_role_and_index(
        SymbolicTrainingLayerPlusContext const &training_layer_plus_context,
        TensorRole tensor_role,
        nonnegative_int index);

std::vector<symbolic_forward_tensor_guid_t>
    get_input_tensors(SymbolicTrainingLayerPlusContext const &);
std::vector<symbolic_gradient_tensor_guid_t>
    get_input_grad_tensors(SymbolicTrainingLayerPlusContext const &);

std::vector<symbolic_forward_tensor_guid_t>
    get_weight_tensors(SymbolicTrainingLayerPlusContext const &);
std::vector<symbolic_gradient_tensor_guid_t>
    get_weight_grad_tensors(SymbolicTrainingLayerPlusContext const &);

std::vector<symbolic_forward_tensor_guid_t>
    get_output_tensors(SymbolicTrainingLayerPlusContext const &);
std::vector<symbolic_gradient_tensor_guid_t>
    get_output_grad_tensors(SymbolicTrainingLayerPlusContext const &);

TrainingLayerSymbolicTensorGroupSignature
    get_tensor_group_signature(SymbolicTrainingLayerPlusContext const &);

} // namespace FlexFlow

#endif
