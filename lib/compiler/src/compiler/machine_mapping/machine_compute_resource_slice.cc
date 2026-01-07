#include "compiler/machine_mapping/machine_compute_resource_slice.h"

namespace FlexFlow {

MachineComputeResourceSlice
    compute_slice_from_specification(MachineComputeSpecification const &spec) {

  return MachineComputeResourceSlice{
      /*num_nodes=*/spec.num_nodes,
      /*num_gpus_per_node=*/spec.num_gpus_per_node,
  };
}

} // namespace FlexFlow
