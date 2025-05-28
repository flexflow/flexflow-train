#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "realm-backend/model_training_instance.h"
#include "utils/containers/reversed.h"

namespace FlexFlow {

  ModelTrainingInstance::ModelTrainingInstance(
    RealmTrainingBacking const &realm_training_backing,
    tensor_guid_t const &logit_tensor,
    loss_tensor_t const &label_tensor,
    LossAttrs const &loss_attrs,
    OptimizerAttrs const &optimizer_attrs)
    : training_backing(realm_training_backing), loss_attrs(loss_attrs),
      optimizer_attrs(optimizer_attrs), logit_tensor(logit_tensor),
      label_tensor(label_tensor){};

PerLayerElapsedTime ModelTrainingInstance::forward() {
  PerLayerElapsedTime per_layer_elapsed_time;
  std::unordered_map<layer_guid_t, Future<float>>
      per_layer_elapsed_time_future;
  for (layer_guid_t const &node : topological_ordering(
           this->training_backing.computation_graph)) {
    per_layer_elapsed_time_future.insert(
        {node, execute_forward(this->training_backing, node)});
  }
  for (layer_guid_t const &node : topological_ordering(
           this->training_backing.computation_graph)) {
    float elapsed_time =
        per_layer_elapsed_time_future[node].get();
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

PerLayerElapsedTime ModelTrainingInstance::backward() {
  compute_loss(this->training_backing,
               this->loss_attrs,
               this->logit_tensor,
               this->label_tensor);

  PerLayerElapsedTime per_layer_elapsed_time;
  std::unordered_map<layer_guid_t, Future<float>>
      per_layer_elapsed_time_future;
  for (layer_guid_t const &node : reversed(topological_ordering(
           this->training_backing.computation_graph))) {
    per_layer_elapsed_time_future.insert(
        {node, execute_backward(this->training_backing, node)});
  }
  for (layer_guid_t const &node : reversed(topological_ordering(
           this->training_backing.computation_graph))) {
    float elapsed_time =
        per_layer_elapsed_time_future[node].get();
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

void ModelTrainingInstance::update() {
  std::unordered_map<layer_guid_t, Future<void>> per_layer_update_future;
  for (layer_guid_t const &node : topological_ordering(
           this->training_backing.computation_graph)) {
    per_layer_update_future.insert(
        {node, execute_update(this->training_backing,
                   node,
                   this->optimizer_attrs)});
  }
  for (layer_guid_t const &node : topological_ordering(
           this->training_backing.computation_graph)) {
    per_layer_update_future[node].wait();
  }
  this->optimizer_attrs = get_optimizer_attrs_for_next_iter(
    this->optimizer_attrs);
}

void ModelTrainingInstance::write_loss_tensor_to_host(float *host_ptr) {
  gradient_tensor_t loss_tensor =
      this->training_backing.realm_tensor_backing
          .tensor_gradient_mapping.at(this->logit_tensor);
  GenericTensorAccessorW loss_tensor_backing =
      this->training_backing.realm_tensor_backing.tensor_backings.at(
          TensorTypeVariant{loss_tensor});
  write_to_host_float_ptr(loss_tensor_backing, host_ptr);
}

} // namespace FlexFlow
