#include "local-execution/model_training_instance.h"
#include "local-execution/execute_task_for_layer.h"
#include "local-execution/local_atomic_tensor_backing.h"
#include "pcg/computation_graph.h"
#include "pcg/optimizer_attrs.h"
#include "task-spec/training_computation_graph.h"
#include "task-spec/training_symbolic_computation_graph.h"
#include "utils/containers/flatmap.h"
#include "utils/containers/reversed.h"
#include "local-execution/local_ready_to_launch_task.dtg.h"

namespace FlexFlow {

ModelTrainingInstance::ModelTrainingInstance(
    Allocator const &allocator,
    LossAttrs const &loss_attrs,
    OptimizerAttrs const &optimizer_attrs)
    : allocator(allocator),
      loss_attrs(loss_attrs), 
      optimizer_attrs(optimizer_attrs) {}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    ModelTrainingInstance::forward() {

  std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
      per_layer_elapsed_time;

  for (symbolic_layer_guid_t symbolic_layer_guid :
       symbolic_cg_topological_ordering(this->symbolic_cg.training_symbolic_computation_graph)) {

    std::optional<milliseconds_t> elapsed_time = execute_forward_for_layer(
      symbolic_layer_guid,
      this->symbolic_cg.training_symbolic_computation_graph,
      this->local_tensor_backing,
      this->local_atomic_tensor_backing,
      this->allocator,
      this->local_task_registry,
      this->runtime_arg_config);

    layer_guid_t layer_guid = this->symbolic_cg.layer_mapping.at_r(symbolic_layer_guid);
    per_layer_elapsed_time.insert({layer_guid, elapsed_time});
  }

  return per_layer_elapsed_time;
}

std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
    ModelTrainingInstance::backward() {
  execute_compute_loss(this->symbolic_cg.training_symbolic_computation_graph,
                       this->local_tensor_backing,
                       this->local_atomic_tensor_backing,
                       this->optimizer_attrs,
                       this->allocator,
                       this->runtime_arg_config);

  std::unordered_map<layer_guid_t, std::optional<milliseconds_t>>
      per_layer_elapsed_time;

  for (symbolic_layer_guid_t symbolic_layer_guid : reversed(symbolic_cg_topological_ordering(
           this->symbolic_cg.training_symbolic_computation_graph))) {

    std::optional<milliseconds_t> elapsed_time = execute_backward_for_layer(
      symbolic_layer_guid,
      this->symbolic_cg.training_symbolic_computation_graph,
      this->local_tensor_backing,
      this->local_atomic_tensor_backing,
      this->allocator,
      this->local_task_registry,
      this->runtime_arg_config);

    layer_guid_t layer_guid = this->symbolic_cg.layer_mapping.at_r(symbolic_layer_guid);
    per_layer_elapsed_time.insert({layer_guid, elapsed_time});
  }
  return per_layer_elapsed_time;
}

void ModelTrainingInstance::update() {
  for (symbolic_layer_guid_t symbolic_layer_guid :
       symbolic_cg_topological_ordering(this->symbolic_cg.training_symbolic_computation_graph)) {
    execute_update_for_layer(symbolic_layer_guid,
                             this->symbolic_cg.training_symbolic_computation_graph,
                             this->local_tensor_backing,
                             this->local_atomic_tensor_backing,
                             this->allocator,
                             this->local_task_registry,
                             this->runtime_arg_config);
  }
  this->optimizer_attrs =
      get_optimizer_attrs_for_next_iter(this->optimizer_attrs);
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
