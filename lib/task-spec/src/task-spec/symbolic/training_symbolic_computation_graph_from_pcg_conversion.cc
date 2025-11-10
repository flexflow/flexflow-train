#include "task-spec/symbolic/training_symbolic_computation_graph_from_pcg_conversion.h"

namespace FlexFlow {

TrainingSymbolicComputationGraphFromPcgConversion generate_training_computation_graph_from_pcg(
    ParallelComputationGraph const &computation_graph,
    OptimizerAttrs const &optimizer_attrs,
    parallel_tensor_guid_t const &logit_tensor,
    SymbolicForwardTensorSource &forward_tensor_source,
    SymbolicGradientTensorSource &gradient_tensor_source,
    SymbolicOptimizerTensorSource &optimizer_tensor_source,
    SymbolicLossTensorSource &loss_tensor_source) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
