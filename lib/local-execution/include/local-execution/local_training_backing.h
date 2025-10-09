#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_LOCAL_TRAINING_BACKING_H

#include "local-execution/local_training_backing.dtg.h"
#include "op-attrs/ops/loss_functions/loss_attrs.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/training_layer_symbolic_tensor_group_signature_with_shapes.dtg.h"
#include "utils/units/milliseconds_t.h"

namespace FlexFlow {

// LocalTrainingBacking make_local_training_backing_for_computation_graph(
//     Allocator &allocator,
//     std::unordered_map<symbolic_training_tensor_guid_t, GenericTensorAccessorW> const
//         &preallocated_tensors,
//     TrainingComputationGraph const &training_computation_graph,
//     RuntimeArgConfig const &runtime_arg_config,
//     OptimizerAttrs const &optimizer_attrs);

} // namespace FlexFlow

#endif
