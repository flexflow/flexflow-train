#include "task-spec/training_layer_tensor_group_signature.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::vector<TrainingTensorGroup> get_training_tensor_groups_for_role(
    TrainingLayerTensorGroupSignature const &signature,
    TensorRole tensor_role) {

  switch (tensor_role) {
    case TensorRole::INPUT:
      return signature.input_tensor_groups;
    case TensorRole::WEIGHT:
      return signature.weight_tensor_groups;
    case TensorRole::OUTPUT:
      return signature.output_tensor_groups;
    default:
      PANIC("Unhandled TensorRole {}", tensor_role);
  }
}

TrainingTensorGroup get_training_tensor_group_for_role_and_index(
    TrainingLayerTensorGroupSignature const &signature,
    TensorRole tensor_role,
    nonnegative_int index) {

  return get_training_tensor_groups_for_role(signature, tensor_role)
      .at(index.unwrap_nonnegative());
}

} // namespace FlexFlow
