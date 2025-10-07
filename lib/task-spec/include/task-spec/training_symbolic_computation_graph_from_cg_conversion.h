#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_SYMBOLIC_COMPUTATION_GRAPH_FROM_CG_CONVERSION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_SYMBOLIC_COMPUTATION_GRAPH_FROM_CG_CONVERSION_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/optimizer_attrs.dtg.h"
#include "task-spec/symbolic_forward_tensor_source.h"
#include "task-spec/symbolic_gradient_tensor_source.h"
#include "task-spec/symbolic_loss_tensor_source.h"
#include "task-spec/symbolic_optimizer_tensor_source.h"
#include "task-spec/training_symbolic_computation_graph_from_cg_conversion.dtg.h"
#include "task-spec/symbolic_training_tensor_group_with_attrs.dtg.h"

namespace FlexFlow {

TrainingSymbolicComputationGraphFromCgConversion generate_training_computation_graph_from_cg(
    ComputationGraph const &computation_graph,
    OptimizerAttrs const &optimizer_attrs,
    tensor_guid_t const &logit_tensor,
    SymbolicForwardTensorSource &forward_tensor_source,
    SymbolicGradientTensorSource &gradient_tensor_source,
    SymbolicOptimizerTensorSource &optimizer_tensor_source,
    SymbolicLossTensorSource &loss_tensor_source);

SymbolicTrainingTensorGroup
    get_training_tensor_group_for_tensor_guid(TrainingSymbolicComputationGraphFromCgConversion const &,
                                              tensor_guid_t);

SymbolicTrainingTensorGroupWithAttrs
    get_training_tensor_group_with_attrs_for_tensor_guid(
        TrainingSymbolicComputationGraphFromCgConversion const &, 
        tensor_guid_t);

symbolic_layer_guid_t
    get_symbolic_layer_guid_for_layer_guid(TrainingSymbolicComputationGraphFromCgConversion const &,
                                           layer_guid_t);

symbolic_tensor_guid_t
    get_symbolic_tensor_guid_for_tensor_guid(TrainingSymbolicComputationGraphFromCgConversion const &,
                                            tensor_guid_t);

tensor_guid_t
    get_tensor_guid_for_symbolic_tensor_guid(TrainingSymbolicComputationGraphFromCgConversion const &,
                                             symbolic_tensor_guid_t);

layer_guid_t
    get_layer_guid_for_symbolic_layer_guid(TrainingSymbolicComputationGraphFromCgConversion const &,
                                           symbolic_layer_guid_t);

} // namespace FlexFlow

#endif
