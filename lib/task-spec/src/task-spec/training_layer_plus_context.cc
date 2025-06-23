#include "task-spec/training_layer_plus_context.h"
#include "task-spec/training_tensor_group_with_attrs.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::vector<TrainingTensorGroupWithAttrs>
    get_training_tensor_groups_with_attrs_for_role(
        TrainingLayerPlusContext const &training_layer_plus_context,
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

TrainingTensorGroupWithAttrs
    get_training_tensor_group_with_attrs_for_role_and_index(
        TrainingLayerPlusContext const &training_layer_plus_context,
        TensorRole tensor_role,
        nonnegative_int index) {

  return get_training_tensor_groups_with_attrs_for_role(
             training_layer_plus_context, tensor_role)
      .at(index.unwrap_nonnegative());
}

std::vector<forward_tensor_guid_t>
    get_input_tensors(TrainingLayerPlusContext const &l) {
  return transform(
      l.input_tensor_groups,
      [](TrainingTensorGroupWithAttrs const &g) { return g.forward_tensor; });
}

std::vector<gradient_tensor_guid_t>
    get_input_grad_tensors(TrainingLayerPlusContext const &l) {
  return transform(
      l.input_tensor_groups,
      [](TrainingTensorGroupWithAttrs const &g) { return g.gradient_tensor; });
}

std::vector<TensorShape>
    get_input_tensor_shapes(TrainingLayerPlusContext const &l) {
  return transform(l.input_tensor_groups,
                   [](TrainingTensorGroupWithAttrs const &g) {
                     return g.tensor_attrs.shape;
                   });
}

std::vector<forward_tensor_guid_t>
    get_weight_tensors(TrainingLayerPlusContext const &l) {
  return transform(
      l.weight_tensor_groups,
      [](TrainingTensorGroupWithAttrs const &g) { return g.forward_tensor; });
}

std::vector<gradient_tensor_guid_t>
    get_weight_grad_tensors(TrainingLayerPlusContext const &l) {
  return transform(
      l.weight_tensor_groups,
      [](TrainingTensorGroupWithAttrs const &g) { return g.gradient_tensor; });
}

std::vector<TensorShape>
    get_weight_tensor_shapes(TrainingLayerPlusContext const &l) {
  return transform(l.weight_tensor_groups,
                   [](TrainingTensorGroupWithAttrs const &g) {
                     return g.tensor_attrs.shape;
                   });
}

std::vector<forward_tensor_guid_t>
    get_output_tensors(TrainingLayerPlusContext const &l) {
  return transform(
      l.output_tensor_groups,
      [](TrainingTensorGroupWithAttrs const &g) { return g.forward_tensor; });
}

std::vector<gradient_tensor_guid_t>
    get_output_grad_tensors(TrainingLayerPlusContext const &l) {
  return transform(
      l.output_tensor_groups,
      [](TrainingTensorGroupWithAttrs const &g) { return g.gradient_tensor; });
}

std::vector<TensorShape>
    get_output_tensor_shapes(TrainingLayerPlusContext const &l) {
  return transform(l.output_tensor_groups,
                   [](TrainingTensorGroupWithAttrs const &g) {
                     return g.tensor_attrs.shape;
                   });
}

TrainingLayerTensorGroupSignature
    get_tensor_group_signature(TrainingLayerPlusContext const &l) {
  return TrainingLayerTensorGroupSignature{
      /*input_tensor_groups=*/transform(l.input_tensor_groups,
                                        tensor_group_without_attrs),
      /*weight_tensor_groups=*/
      transform(l.weight_tensor_groups, tensor_group_without_attrs),
      /*output_tensor_groups=*/
      transform(l.output_tensor_groups, tensor_group_without_attrs),
  };
}

CGOperatorTensorShapeSignature
    get_cg_op_shape_signature(TrainingLayerPlusContext const &l) {
  return CGOperatorTensorShapeSignature{
      /*input_shapes=*/get_input_tensor_shapes(l),
      /*weight_shapes=*/get_weight_tensor_shapes(l),
      /*output_shapes=*/get_output_tensor_shapes(l),
  };
}

} // namespace FlexFlow
