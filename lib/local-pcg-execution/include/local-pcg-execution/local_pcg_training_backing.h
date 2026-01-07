#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PCG_TRAINING_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PCG_TRAINING_BACKING_H

#include "local-pcg-execution/local_pcg_training_backing.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/training_parallel_layer_plus_context.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

LocalPcgTrainingBacking make_local_pcg_training_backing_for_pcg(
    Allocator &allocator,
    std::unordered_map<training_parallel_tensor_guid_t,
                       ParallelTensorAccessorsW> const &preallocated_tensors,
    TrainingParallelComputationGraph const &training_pcg,
    RuntimeArgConfig const &runtime_arg_config,
    OptimizerAttrs const &optimizer_attrs,
    MachineComputeSpecification const &machine_compute_specification);

std::optional<std::unordered_map<gpu_id_t, milliseconds_t>>
    execute_forward(LocalTaskRegistry const &,
                    LocalParallelTensorBacking const &,
                    LocalPcgArgsBacking const &,
                    TrainingParallelLayerPlusContext const &,
                    Allocator &);

std::optional<std::unordered_map<gpu_id_t, milliseconds_t>> execute_backward();

void compute_loss(LocalPcgTrainingBacking const &,
                  LossAttrs const &,
                  Allocator &);

void execute_update(LocalPcgTrainingBacking const &,
                    parallel_layer_guid_t const &,
                    OptimizerAttrs const &,
                    Allocator &);

} // namespace FlexFlow

#endif
