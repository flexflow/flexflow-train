#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_communication_edge.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_device.h"
#include "pcg/machine_compute_specification.dtg.h"

namespace FlexFlow {

CommunicationEdge
  concretize_abstracted_communication_edge(
    AbstractedCommunicationEdge const &edge,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &src_machine_stencils,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &dst_machine_stencils) {

  return CommunicationEdge{
    /*src=*/concretize_abstracted_device(edge.src, src_machine_stencils),
    /*dst=*/concretize_abstracted_device(edge.dst, dst_machine_stencils),
  };
}

} // namespace FlexFlow
