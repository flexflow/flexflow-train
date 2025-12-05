#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "realm-backend/model_training_instance.h"
#include "task-spec/training_computation_graph.h"
#include "utils/containers/reversed.h"

namespace FlexFlow {

  ModelTrainingInstance::ModelTrainingInstance(
    RealmRuntimeState &runtime_state,
    LocalTrainingBacking const &local_training_backing,
    LossAttrs const &loss_attrs,
    OptimizerAttrs const &optimizer_attrs)
    : runtime_state(runtime_state), training_backing(local_training_backing), 
      loss_attrs(loss_attrs), optimizer_attrs(optimizer_attrs) {}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    ModelTrainingInstance::forward() {

  std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
      per_layer_elapsed_time;
    std::unordered_map<layer_guid_t, Future<std::optional<milliseconds_t>>>
      per_layer_elapsed_time_future;

  for (layer_guid_t const &layer_guid :
       topological_ordering(this->training_backing.training_computation_graph
                                .computation_graph)) {
    per_layer_elapsed_time_future.insert(
        {layer_guid, 
         execute_forward(
          this->training_backing.local_task_registry,
          this->training_backing.local_tensor_backing,
          this->training_backing.local_args_backing,
          get_training_layer_plus_context(
              this->training_backing.training_computation_graph, layer_guid),
          this->runtime_state)
        });
  }

  for (layer_guid_t const &layer_guid : topological_ordering(
           this->training_backing.training_computation_graph
               .computation_graph)) {
    std::optional<milliseconds_t> elapsed_time =
        per_layer_elapsed_time_future[layer_guid].get();
    per_layer_elapsed_time.insert({layer_guid, elapsed_time});
  }
  return per_layer_elapsed_time;
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    ModelTrainingInstance::backward() {
  compute_loss(this->training_backing, this->loss_attrs, this->runtime_state);

  std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
      per_layer_elapsed_time;
  std::unordered_map<layer_guid_t, Future<std::optional<milliseconds_t>>>
      per_layer_elapsed_time_future;

  for (layer_guid_t const &layer_guid : reversed(topological_ordering(
           this->training_backing.training_computation_graph
               .computation_graph))) {
    per_layer_elapsed_time_future.insert(
        {layer_guid, 
         execute_backward(
          this->training_backing.local_task_registry,
          this->training_backing.local_tensor_backing,
          this->training_backing.local_args_backing,
          get_training_layer_plus_context(
              this->training_backing.training_computation_graph, layer_guid),
          this->runtime_state)
        });
  }
  
  for (layer_guid_t const &layer_guid : reversed(topological_ordering(
           this->training_backing.training_computation_graph
               .computation_graph))) {
    std::optional<milliseconds_t> elapsed_time =
        per_layer_elapsed_time_future[layer_guid].get();
    per_layer_elapsed_time.insert({layer_guid, elapsed_time});
  }
  return per_layer_elapsed_time;
}

void ModelTrainingInstance::update() {
  std::unordered_map<layer_guid_t, Future<void>> per_layer_update_future;
  for (layer_guid_t const &layer_guid : topological_ordering(
           this->training_backing.training_computation_graph
               .computation_graph)) {
    per_layer_update_future.insert(
        {layer_guid, execute_update(this->training_backing,
                   layer_guid,
                   this->optimizer_attrs,
                   this->runtime_state)});
  }
  for (layer_guid_t const &layer_guid : topological_ordering(
           this->training_backing.training_computation_graph
               .computation_graph)) {
    per_layer_update_future[layer_guid].wait();
  }
  this->optimizer_attrs = get_optimizer_attrs_for_next_iter(
    this->optimizer_attrs);
}

GenericTensorAccessorR ModelTrainingInstance::get_loss_tensor_accessor() const {
  gradient_tensor_guid_t loss_tensor = get_gradient_tensor_guid_for_tensor_guid(
      this->training_backing.training_computation_graph,
      this->training_backing.training_computation_graph.logit_tensor);
  GenericTensorAccessorW loss_tensor_backing =
      this->training_backing.local_tensor_backing
          .backing_for_training_tensor_map.at(
              training_tensor_guid_t{loss_tensor});
  return read_only_accessor_from_write_accessor(loss_tensor_backing);
}

} // namespace FlexFlow
