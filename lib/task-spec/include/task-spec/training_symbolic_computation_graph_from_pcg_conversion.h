#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_SYMBOLIC_COMPUTATION_GRAPH_FROM_PCG_CONVERSION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TRAINING_SYMBOLIC_COMPUTATION_GRAPH_FROM_PCG_CONVERSION_H

#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.dtg.h"
#include "task-spec/symbolic_forward_tensor_source.h"
#include "task-spec/symbolic_gradient_tensor_source.h"
#include "task-spec/symbolic_loss_tensor_source.h"
#include "task-spec/symbolic_optimizer_tensor_source.h"
#include "task-spec/training_symbolic_computation_graph_from_pcg_conversion.dtg.h"


namespace FlexFlow {

TrainingSymbolicComputationGraphFromPcgConversion generate_training_computation_graph_from_pcg(
    ParallelComputationGraph const &computation_graph,
    OptimizerAttrs const &optimizer_attrs,
    parallel_tensor_guid_t const &logit_tensor,
    SymbolicForwardTensorSource &forward_tensor_source,
    SymbolicGradientTensorSource &gradient_tensor_source,
    SymbolicOptimizerTensorSource &optimizer_tensor_source,
    SymbolicLossTensorSource &loss_tensor_source);

} // namespace FlexFlow

#endif
