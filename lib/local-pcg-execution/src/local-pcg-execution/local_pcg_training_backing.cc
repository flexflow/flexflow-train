#include "local-pcg-execution/local_pcg_training_backing.h"
#include "local-execution/local_task_registry.h"

namespace FlexFlow {

LocalPcgTrainingBacking make_local_pcg_training_backing_for_pcg(
  Allocator &allocator,
  std::unordered_map<training_parallel_tensor_guid_t, ParallelTensorAccessorsW>
    const &preallocated_tensors,
  TrainingParallelComputationGraph const &training_pcg,
  RuntimeArgConfig const &runtime_arg_config,
  OptimizerAttrs const &optimizer_attrs,
  MachineComputeSpecification const &machine_compute_specification) {

  NOT_IMPLEMENTED();
}

std::optional<std::unordered_map<gpu_id_t, milliseconds_t>>
  execute_forward(LocalTaskRegistry const &local_task_registry,
                  LocalParallelTensorBacking const &,
                  LocalPcgArgsBacking const &,
                  TrainingParallelLayerPlusContext const &training_parallel_layer,
                  Allocator &) {
  
  // std::optional maybe_registered_task = try_get_registered_task(
  //     local_task_registry, training_parallel_layer.parallel_layer_guid, OpTaskType::FWD);
  //
  // ASSERT(maybe_registered_task.has_value());
  //
  // registered_task_t registered_task = maybe_registered_task.value();
  // if (registered_task.is_noop_task()) {
  //   return std::nullopt;
  // }

  NOT_IMPLEMENTED();
}

std::optional<std::unordered_map<gpu_id_t, milliseconds_t>>
  execute_backward() {
  NOT_IMPLEMENTED();
}

void compute_loss(LocalPcgTrainingBacking const &, LossAttrs const &, Allocator &) {
  NOT_IMPLEMENTED();
}

void execute_update(LocalPcgTrainingBacking const &,
                    parallel_layer_guid_t const &,
                    OptimizerAttrs const &,
                    Allocator &) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
