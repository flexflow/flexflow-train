#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_LAYER_PLUS_CONTEXT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_LAYER_PLUS_CONTEXT_H

#include "pcg/cg_operator_tensor_shape_signature.dtg.h"
#include "pcg/tensor_role.dtg.h"
#include "task-spec/training_layer_plus_context.dtg.h"
#include "task-spec/training_layer_tensor_group_signature.dtg.h"

namespace FlexFlow {

std::vector<TrainingTensorGroupWithAttrs>
    get_training_tensor_groups_with_attrs_for_role(
        TrainingLayerPlusContext const &training_layer_plus_context,
        TensorRole tensor_role);

TrainingTensorGroupWithAttrs
    get_training_tensor_group_with_attrs_for_role_and_index(
        TrainingLayerPlusContext const &training_layer_plus_context,
        TensorRole tensor_role,
        nonnegative_int index);

std::vector<forward_tensor_guid_t>
    get_input_tensors(TrainingLayerPlusContext const &);
std::vector<gradient_tensor_guid_t>
    get_input_grad_tensors(TrainingLayerPlusContext const &);
std::vector<TensorShape>
    get_input_tensor_shapes(TrainingLayerPlusContext const &);

std::vector<forward_tensor_guid_t>
    get_weight_tensors(TrainingLayerPlusContext const &);
std::vector<gradient_tensor_guid_t>
    get_weight_grad_tensors(TrainingLayerPlusContext const &);
std::vector<TensorShape>
    get_weight_tensor_shapes(TrainingLayerPlusContext const &);

std::vector<forward_tensor_guid_t>
    get_output_tensors(TrainingLayerPlusContext const &);
std::vector<gradient_tensor_guid_t>
    get_output_grad_tensors(TrainingLayerPlusContext const &);
std::vector<TensorShape>
    get_output_tensor_shapes(TrainingLayerPlusContext const &);

TrainingLayerTensorGroupSignature
    get_tensor_group_signature(TrainingLayerPlusContext const &);
CGOperatorTensorShapeSignature
    get_cg_op_shape_signature(TrainingLayerPlusContext const &);

} // namespace FlexFlow

#endif
