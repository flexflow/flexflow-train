#include "pcg/machine_compute_resource_slice.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

MachineComputeResourceSlice
    compute_slice_from_specification(MachineComputeSpecification const &spec) {

  return MachineComputeResourceSlice{
      /*num_nodes=*/spec.num_nodes,
      /*num_gpus_per_node=*/spec.num_gpus_per_node,
  };
}

positive_int
    get_total_num_devices_in_slice(MachineComputeResourceSlice const &slice) {
  return slice.num_nodes * slice.num_gpus_per_node;
}

bool is_valid_machine_space_coordinate_in_slice(
    MachineComputeResourceSlice const &slice,
    MachineSpaceCoordinate const &coord) {
  ASSERT(coord.device_type == DeviceType::GPU);

  return (coord.node_idx < slice.num_nodes) &&
         (coord.device_idx < slice.num_gpus_per_node);
}

} // namespace FlexFlow
