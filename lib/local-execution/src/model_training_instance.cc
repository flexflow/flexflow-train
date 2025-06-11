#include "local-execution/model_training_instance.h"
#include "kernels/format_accessor_contents.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "utils/containers/reversed.h"

namespace FlexFlow {

ModelTrainingInstance::ModelTrainingInstance(
    Allocator const &allocator,
    LocalTrainingBacking const &local_training_backing,
    tensor_guid_t const &logit_tensor,
    loss_tensor_t const &label_tensor,
    LossAttrs const &loss_attrs,
    OptimizerAttrs const &optimizer_attrs)
    : allocator(allocator), training_backing(local_training_backing),
      loss_attrs(loss_attrs), optimizer_attrs(optimizer_attrs),
      logit_tensor(logit_tensor), label_tensor(label_tensor){};

PerLayerElapsedTime ModelTrainingInstance::forward() {
  PerLayerElapsedTime per_layer_elapsed_time;
  for (layer_guid_t const &node :
       topological_ordering(this->training_backing.computation_graph)) {
    std::optional<float> elapsed_time =
        execute_forward(this->training_backing, node, this->allocator);
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

PerLayerElapsedTime ModelTrainingInstance::backward() {
  compute_loss(this->training_backing,
               this->loss_attrs,
               this->logit_tensor,
               this->label_tensor,
               this->allocator);

  std::cout << "Done computing loss" << std::endl;
  gradient_tensor_t loss_tensor =
      this->training_backing.local_tensor_backing.tensor_gradient_mapping.at(
          this->logit_tensor);
  GenericTensorAccessorW loss_tensor_backing =
      this->training_backing.local_tensor_backing.tensor_backings.at(
          TensorTypeVariant{loss_tensor});
  
  std::cout << "Loss (logit grad) tensor" << std::endl;
  std::cout << format_accessor_w_contents(loss_tensor_backing) << std::endl;

  PerLayerElapsedTime per_layer_elapsed_time;
  for (layer_guid_t const &node : reversed(
           topological_ordering(this->training_backing.computation_graph))) {
    std::optional<float> elapsed_time =
        execute_backward(this->training_backing, node, this->allocator);
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

void ModelTrainingInstance::update() {
  for (layer_guid_t const &node :
       topological_ordering(this->training_backing.computation_graph)) {
    execute_update(
        this->training_backing, node, this->optimizer_attrs, this->allocator);
  }
  this->optimizer_attrs =
      get_optimizer_attrs_for_next_iter(this->optimizer_attrs);
}

GenericTensorAccessorR ModelTrainingInstance::get_loss_tensor_accessor() const {
  GenericTensorAccessorW logit_tensor_backing = this->training_backing
      .local_tensor_backing.tensor_backings.at(TensorTypeVariant{this->logit_tensor});

  // for (auto const &pair :
  //      this->training_backing.local_tensor_backing.tensor_backings) {
  //   std::cout << "Tensor type: " << pair.first << std::endl;
  //   std::cout << "Tensor " << std::endl;
  //   std::cout << format_accessor_w_contents(pair.second) << std::endl;
  // }

  gradient_tensor_t loss_tensor =
      this->training_backing.local_tensor_backing.tensor_gradient_mapping.at(
          this->logit_tensor);
  GenericTensorAccessorW loss_tensor_backing =
      this->training_backing.local_tensor_backing.tensor_backings.at(
          TensorTypeVariant{loss_tensor});
  
  std::cout << "Loss (logit grad) tensor" << std::endl;
  std::cout << format_accessor_w_contents(loss_tensor_backing) << std::endl;
  return read_only_accessor_from_write_accessor(loss_tensor_backing);
}

} // namespace FlexFlow
