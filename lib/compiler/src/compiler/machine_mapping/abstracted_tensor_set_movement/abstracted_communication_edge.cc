#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_communication_edge.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_device.h"
#include "pcg/machine_compute_specification.dtg.h"

namespace FlexFlow {

CommunicationEdge
  concretize_abstracted_communication_edge(
    AbstractedCommunicationEdge const &edge,
    std::unordered_map<BinaryTreePath, OperatorTaskSpace> const &src_task_spaces,
    ParallelLayerGuidObliviousMachineMapping const &src_mapping,
    std::unordered_map<BinaryTreePath, OperatorTaskSpace> const &dst_task_spaces,
    ParallelLayerGuidObliviousMachineMapping const &dst_mapping) {

  return CommunicationEdge{
    /*src=*/concretize_abstracted_device(edge.src, src_task_spaces, src_mapping),
    /*dst=*/concretize_abstracted_device(edge.dst, dst_task_spaces, dst_mapping),
  };
}

} // namespace FlexFlow
