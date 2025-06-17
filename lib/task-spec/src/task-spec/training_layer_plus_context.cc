#include "task-spec/training_layer_plus_context.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

std::vector<forward_tensor_guid_t> get_input_tensors(TrainingLayerPlusContext const &l) {
  return transform(l.input_tensor_groups,
                   [](TrainingTensorGroupWithAttrs const &g) {
                     return g.forward_tensor;
                   });
}

std::vector<gradient_tensor_guid_t> get_input_grad_tensors(TrainingLayerPlusContext const &l) {
  return transform(l.input_tensor_groups,
                   [](TrainingTensorGroupWithAttrs const &g) {
                     return g.gradient_tensor;
                   });
}

std::vector<TensorShape> get_input_tensor_shapes(TrainingLayerPlusContext const &l) {
  return transform(l.input_tensor_groups,
                   [](TrainingTensorGroupWithAttrs const &g) {
                     return g.tensor_attrs.shape;
                   });
}

std::vector<forward_tensor_guid_t> get_weight_tensors(TrainingLayerPlusContext const &l) {
  return transform(l.weight_tensor_groups,
                   [](TrainingTensorGroupWithAttrs const &g) {
                     return g.forward_tensor;
                   });
}


std::vector<gradient_tensor_guid_t> get_weight_grad_tensors(TrainingLayerPlusContext const &l) {
  return transform(l.weight_tensor_groups,
                   [](TrainingTensorGroupWithAttrs const &g) {
                     return g.gradient_tensor;
                   });
}

std::vector<forward_tensor_guid_t> get_output_tensors(TrainingLayerPlusContext const &l) {
  return transform(l.output_tensor_groups,
                   [](TrainingTensorGroupWithAttrs const &g) {
                     return g.forward_tensor;
                   });
}

std::vector<gradient_tensor_guid_t> get_output_grad_tensors(TrainingLayerPlusContext const &l) {
  return transform(l.output_tensor_groups,
                   [](TrainingTensorGroupWithAttrs const &g) {
                     return g.gradient_tensor;
                   });
}

} // namespace FlexFlow
