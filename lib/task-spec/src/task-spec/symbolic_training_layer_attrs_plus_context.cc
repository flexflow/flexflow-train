#include "task-spec/symbolic_training_layer_attrs_plus_context.h"
#include "utils/containers/transform.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

std::vector<SymbolicTrainingTensorGroup>
    get_training_tensor_groups_for_role(
        SymbolicTrainingLayerAttrsPlusContext const &training_layer_plus_context,
        TensorRole tensor_role) {

  switch (tensor_role) {
    case TensorRole::INPUT:
      return training_layer_plus_context.input_tensor_groups;
    case TensorRole::WEIGHT:
      return training_layer_plus_context.weight_tensor_groups;
    case TensorRole::OUTPUT:
      return training_layer_plus_context.output_tensor_groups;
    default:
      PANIC("Unhandled TensorRole {}", tensor_role);
  }
}

SymbolicTrainingTensorGroup
    get_training_tensor_group_for_role_and_index(
        SymbolicTrainingLayerAttrsPlusContext const &training_layer_plus_context,
        TensorRole tensor_role,
        nonnegative_int index) {

  return get_training_tensor_groups_for_role(
             training_layer_plus_context, tensor_role)
      .at(index.unwrap_nonnegative());
}

std::vector<symbolic_forward_tensor_guid_t>
    get_input_tensors(SymbolicTrainingLayerAttrsPlusContext const &l) {
  return transform(
      l.input_tensor_groups,
      [](SymbolicTrainingTensorGroup const &g) { return g.forward_tensor; });
}

std::vector<symbolic_gradient_tensor_guid_t>
    get_input_grad_tensors(SymbolicTrainingLayerAttrsPlusContext const &l) {
  return transform(
      l.input_tensor_groups,
      [](SymbolicTrainingTensorGroup const &g) { return g.gradient_tensor; });
}

std::vector<symbolic_forward_tensor_guid_t>
    get_weight_tensors(SymbolicTrainingLayerAttrsPlusContext const &l) {
  return transform(
      l.weight_tensor_groups,
      [](SymbolicTrainingTensorGroup const &g) { return g.forward_tensor; });
}

std::vector<symbolic_gradient_tensor_guid_t>
    get_weight_grad_tensors(SymbolicTrainingLayerAttrsPlusContext const &l) {
  return transform(
      l.weight_tensor_groups,
      [](SymbolicTrainingTensorGroup const &g) { return g.gradient_tensor; });
}

std::vector<symbolic_forward_tensor_guid_t>
    get_output_tensors(SymbolicTrainingLayerAttrsPlusContext const &l) {
  return transform(
      l.output_tensor_groups,
      [](SymbolicTrainingTensorGroup const &g) { return g.forward_tensor; });
}

std::vector<symbolic_gradient_tensor_guid_t>
    get_output_grad_tensors(SymbolicTrainingLayerAttrsPlusContext const &l) {
  return transform(
      l.output_tensor_groups,
      [](SymbolicTrainingTensorGroup const &g) { return g.gradient_tensor; });
}

TrainingLayerSymbolicTensorGroupSignature
    get_tensor_group_signature(SymbolicTrainingLayerAttrsPlusContext const &l) {
  return TrainingLayerSymbolicTensorGroupSignature{
      /*input_tensor_groups=*/l.input_tensor_groups,
      /*weight_tensor_groups=*/l.weight_tensor_groups,
      /*output_tensor_groups=*/l.output_tensor_groups,
  };
}

} // namespace FlexFlow
