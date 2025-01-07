#include "local-execution/model_training_instance.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "utils/containers/reversed.h"

namespace FlexFlow {

ModelTrainingInstance::ModelTrainingInstance(
    Allocator const &allocator,
    ComputationGraph const &computation_graph,
    LayerTensorBackingMap const &allocated_forward_tensors,
    TensorBackingMap const &allocated_non_graph_tensors,
    RuntimeArgConfig const &runtime_arg_config,
    LossAttrs const &loss_attrs,
    reduced_tensor_t const &logit_tensor,
    reduced_tensor_t const &label_tensor,
    OptimizerAttrs const &optimizer_attrs)
    : computation_graph(computation_graph),
      training_backing(allocator,
                       computation_graph,
                       allocated_forward_tensors,
                       allocated_non_graph_tensors,
                       runtime_arg_config),
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
  for (layer_guid_t const &node :
       topological_ordering(this->computation_graph)) {
    std::optional<float> elapsed_time =
        this->training_backing.execute_forward(node);
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

PerLayerElapsedTime ModelTrainingInstance::execute_backward() {
  this->training_backing.compute_loss(
      this->loss_attrs, this->logit_tensor, this->label_tensor);

  PerLayerElapsedTime per_layer_elapsed_time;
  for (layer_guid_t const &node :
       reversed(topological_ordering(this->computation_graph))) {
    std::optional<float> elapsed_time =
        this->training_backing.execute_backward(node);
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

void ModelTrainingInstance::execute_update() {
  for (layer_guid_t const &node :
       topological_ordering(this->computation_graph)) {
    this->training_backing.execute_update(node, this->optimizer_attrs);
  }
  this->optimizer_attrs =
      get_optimizer_attrs_for_next_iter(this->optimizer_attrs);
}

} // namespace FlexFlow
