#include "compiler/cost_estimator/tensor_set_movement.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
namespace FlexFlow {

TensorSetMovement get_tensor_set_movement_from_pcg_edge(
    ParallelComputationGraphEdge const &edge,
    ParallelComputationGraph const &pcg,
    MachineView const &src_mv,
    MachineView const &dst_mv) {
  ParallelTensorShape tensor_shape =
      get_parallel_tensor_shape(pcg, parallel_tensor_guid_t{edge.raw_edge.src});
  return TensorSetMovement{
      {SingleTensorMovement{tensor_shape, {src_mv}, {dst_mv}}}};
}

} // namespace FlexFlow
