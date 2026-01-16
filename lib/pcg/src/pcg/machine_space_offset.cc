#include "pcg/machine_space_offset.h"
#include "utils/exception.h"

namespace FlexFlow {

MachineSpaceOffset get_machine_space_offset_from_coordinate(
    MachineSpaceCoordinate const &start, MachineSpaceCoordinate const &coord) {
  ASSERT(start.device_idx <= coord.device_idx,
         "The start device_idx is greater than one of the coord device_idx."
         "Are you sure you didn't swap them?");

  ASSERT(start.node_idx <= coord.device_idx,
         "The start node_idx is greater than one of the coord node_idx."
         "Are you sure you didn't swap them?");

  ASSERT(start.device_type == coord.device_type);

  return MachineSpaceOffset{
      /*node_offset=*/coord.node_idx.unwrap_nonnegative() -
          start.node_idx.unwrap_nonnegative(),
      /*device_offset=*/coord.device_idx.unwrap_nonnegative() -
          start.device_idx.unwrap_nonnegative(),
      /*device_type=*/coord.device_type,
  };
}

} // namespace FlexFlow
