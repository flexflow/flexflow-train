#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H

#include "local-execution/local_training_backing.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/training_computation_graph.dtg.h"
#include "task-spec/training_tensor_guid_t.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

LocalTrainingBacking make_local_training_backing_for_computation_graph(
    Allocator &allocator,
    std::unordered_map<training_tensor_guid_t, GenericTensorAccessorW> const
        &preallocated_tensors,
    TrainingComputationGraph const &training_computation_graph,
    RuntimeArgConfig const &runtime_arg_config,
    OptimizerAttrs const &optimizer_attrs);

std::optional<milliseconds_t> execute_forward(LocalTaskRegistry const &,
                                              LocalTensorBacking const &,
                                              LocalArgsBacking const &,
                                              TrainingLayerPlusContext const &,
                                              Allocator &);

std::optional<milliseconds_t> execute_backward(LocalTaskRegistry const &,
                                               LocalTensorBacking const &,
                                               LocalArgsBacking const &,
                                               TrainingLayerPlusContext const &,
                                               Allocator &);

void compute_loss(LocalTrainingBacking const &,
                  LossAttrs const &,
                  Allocator &);

void execute_update(LocalTrainingBacking const &,
                    layer_guid_t const &,
                    OptimizerAttrs const &,
                    Allocator &);

} // namespace FlexFlow

#endif
