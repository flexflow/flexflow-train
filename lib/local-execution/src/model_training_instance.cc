#include "local-execution/model_training_instance.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "utils/containers/reversed.h"

namespace FlexFlow {

ModelTrainingInstance::ModelTrainingInstance(
    LocalTrainingBacking const &local_training_backing,
    tensor_guid_t const & logit_tensor,
    TensorShape const &label_tensor_shape,
    LossAttrs const &loss_attrs,
    OptimizerAttrs const &optimizer_attrs)
    : training_backing(local_training_backing), loss_attrs(loss_attrs),
      optimizer_attrs(optimizer_attrs), logit_tensor(logit_tensor),
      label_tensor(
          allocate_loss_tensor(this->training_backing.local_tensor_backing,
                               this->loss_tensor_source,
                               label_tensor_shape,
                               this->training_backing.allocator)) {
  allocate_all_optimizer_tensors(this->training_backing.local_tensor_backing,
                                 this->optimizer_tensor_source,
                                 this->training_backing.computation_graph,
                                 this->training_backing.allocator,
                                 this->optimizer_attrs);
}

void init(ModelTrainingInstance &model_training_instance) {
  for (layer_guid_t const &node : topological_ordering(
           model_training_instance.training_backing.computation_graph)) {
    execute_init(model_training_instance.training_backing, node);
  }
}

PerLayerElapsedTime forward(ModelTrainingInstance &model_training_instance) {
  PerLayerElapsedTime per_layer_elapsed_time;
  for (layer_guid_t const &node : topological_ordering(
           model_training_instance.training_backing.computation_graph)) {
    std::optional<float> elapsed_time =
        execute_forward(model_training_instance.training_backing, node);
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

PerLayerElapsedTime backward(ModelTrainingInstance &model_training_instance) {
  compute_loss(model_training_instance.training_backing,
               model_training_instance.loss_attrs, 
               model_training_instance.logit_tensor,
               model_training_instance.label_tensor);

  PerLayerElapsedTime per_layer_elapsed_time;
  for (layer_guid_t const &node : reversed(topological_ordering(
           model_training_instance.training_backing.computation_graph))) {
    std::optional<float> elapsed_time =
        execute_backward(model_training_instance.training_backing, node);
    per_layer_elapsed_time.insert({node, elapsed_time});
  }
  return per_layer_elapsed_time;
}

void update(ModelTrainingInstance & model_training_instance) {
  for (layer_guid_t const &node :
       topological_ordering(model_training_instance.training_backing.computation_graph)) {
    execute_update(model_training_instance.training_backing, node, model_training_instance.optimizer_attrs);
  }
  model_training_instance.optimizer_attrs =
      get_optimizer_attrs_for_next_iter(model_training_instance.optimizer_attrs);
}

} // namespace FlexFlow
