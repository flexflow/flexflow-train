#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_LAYER_PLUS_CONTEXT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_LAYER_PLUS_CONTEXT_H

#include "task-spec/training_layer_plus_context.dtg.h"

namespace FlexFlow {

std::vector<forward_tensor_guid_t> get_input_tensors(TrainingLayerPlusContext const &);
std::vector<gradient_tensor_guid_t> get_input_grad_tensors(TrainingLayerPlusContext const &);
std::vector<TensorShape> get_input_tensor_shapes(TrainingLayerPlusContext const &);

std::vector<forward_tensor_guid_t> get_weight_tensors(TrainingLayerPlusContext const &);
std::vector<gradient_tensor_guid_t> get_weight_grad_tensors(TrainingLayerPlusContext const &);

std::vector<forward_tensor_guid_t> get_output_tensors(TrainingLayerPlusContext const &);
std::vector<gradient_tensor_guid_t> get_output_grad_tensors(TrainingLayerPlusContext const &);

} // namespace FlexFlow

#endif
