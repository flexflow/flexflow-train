#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_single_tensor_communication_edge.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_device.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/machine_space_stencil.h"
#include "pcg/machine_compute_specification.dtg.h"

namespace FlexFlow {

std::optional<CommunicationEdge>
  concretize_abstracted_single_tensor_communication_edge(
    AbstractedSingleTensorCommunicationEdge const &edge,
    MachineSpaceStencil const &src_machine_stencil,
    std::unordered_map<BinaryTreePath, MachineSpaceStencil> const &dst_machine_stencils) {

  MachineSpaceCoordinate src = machine_space_stencil_compute_machine_coord(src_machine_stencil, edge.src_coord);
  MachineSpaceCoordinate dst = concretize_abstracted_device(edge.dst, dst_machine_stencils);

  if (src == dst) {
    return std::nullopt;
  } else {
    return CommunicationEdge{
      /*src=*/src,
      /*dst=*/dst,
    };
  }
}

} // namespace FlexFlow
