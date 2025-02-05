#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "realm-backend/model_training_instance.h"
#include "utils/containers/reversed.h"

namespace FlexFlow {

ModelTrainingInstance::ModelTrainingInstance(
    ComputationGraph const &computation_graph,
    RuntimeArgConfig const &runtime_arg_config, LossAttrs const &loss_attrs,
    tensor_guid_t const &logit_tensor, loss_tensor_t const &label_tensor,
    OptimizerAttrs const &optimizer_attrs)
    : computation_graph(computation_graph),
      training_backing(computation_graph, runtime_arg_config),
      loss_attrs(loss_attrs), logit_tensor(logit_tensor),
      label_tensor(label_tensor), optimizer_attrs(optimizer_attrs) {

  // allocate each layer's tensors
  for (layer_guid_t const &node :
       topological_ordering(this->computation_graph)) {
    this->training_backing.register_and_allocate_layer(node);
    this->training_backing.allocate_layer_optimizer_tensors(
        node, this->optimizer_attrs);
  }
}

void ModelTrainingInstance::execute_init() {
  for (layer_guid_t const &node :
       topological_ordering(this->computation_graph)) {
    this->training_backing.execute_init(node);
  }
}

PerLayerElapsedTime ModelTrainingInstance::execute_forward() {
  PerLayerElapsedTime per_layer_elapsed_time;
  std::unordered_map<layer_guid_t, Future<std::optional<float>>>
      per_layer_elapsed_time_future;
  for (layer_guid_t const &node :
       topological_ordering(this->computation_graph)) {
    per_layer_elapsed_time_future.insert(
        {node, this->training_backing.execute_forward(node)});
  }
  for (layer_guid_t const &node :
       topological_ordering(this->computation_graph)) {
    std::optional<float> elapsed_time =
        per_layer_elapsed_time_future[node].get();
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

PerLayerElapsedTime ModelTrainingInstance::execute_backward() {
  this->training_backing.compute_loss(this->loss_attrs, this->logit_tensor,
                                      this->label_tensor);
  PerLayerElapsedTime per_layer_elapsed_time;
  std::unordered_map<layer_guid_t, Future<std::optional<float>>>
      per_layer_elapsed_time_future;
  for (layer_guid_t const &node :
       reversed(topological_ordering(this->computation_graph))) {
    per_layer_elapsed_time_future.insert(
        {node, this->training_backing.execute_backward(node)});
  }
  for (layer_guid_t const &node :
       reversed(topological_ordering(this->computation_graph))) {
    std::optional<float> elapsed_time =
        per_layer_elapsed_time_future[node].get();
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

void ModelTrainingInstance::execute_update() {
  std::unordered_map<layer_guid_t, Future<void>> per_layer_update_future;
  for (layer_guid_t const &node :
       topological_ordering(this->computation_graph)) {
    per_layer_update_future.insert(
        {node, this->training_backing.execute_update(node, this->optimizer_attrs)});
  }
  for (layer_guid_t const &node :
       topological_ordering(this->computation_graph)) {
    per_layer_update_future[node].wait();
  }
  this->optimizer_attrs =
      get_optimizer_attrs_for_next_iter(this->optimizer_attrs);
}

} // namespace FlexFlow
