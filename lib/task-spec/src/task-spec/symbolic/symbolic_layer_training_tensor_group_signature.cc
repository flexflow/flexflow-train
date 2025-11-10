#include "task-spec/symbolic/symbolic_layer_training_tensor_group_signature.h"
#include "task-spec/symbolic/symbolic_training_tensor_group.h"
#include <libassert/assert.hpp>
#include "utils/containers/transform.h"

namespace FlexFlow {

std::vector<SymbolicTrainingTensorGroup> get_training_tensor_groups_for_role(
    SymbolicLayerTrainingTensorGroupSignature const &signature,
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

SymbolicTrainingTensorGroup get_training_tensor_group_for_role_and_index(
    SymbolicLayerTrainingTensorGroupSignature const &signature,
    TensorRole tensor_role,
    nonnegative_int index) {

  return get_training_tensor_groups_for_role(signature, tensor_role)
      .at(index.unwrap_nonnegative());
}

std::vector<symbolic_training_tensor_guid_t>
  get_training_tensors_for_role_and_type(SymbolicLayerTrainingTensorGroupSignature const &signature,
                                         TensorRole tensor_role,
                                         FwbTensorType tensor_type) {
  std::vector<SymbolicTrainingTensorGroup>
    groups = get_training_tensor_groups_for_role(signature, tensor_role);

  return transform(groups, 
                   [&](SymbolicTrainingTensorGroup const &g) -> symbolic_training_tensor_guid_t {
                     return get_training_tensor_for_type(g, tensor_type);
                   });
}


} // namespace FlexFlow
