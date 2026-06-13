#include "task-spec/global_device_id_t.h"

namespace FlexFlow {

global_device_id_t global_device_id_from_local(
  local_device_id_t const &local_device_id,
  node_idx_t node_idx)
{
  return global_device_id_t{
    /*coord=*/MachineSpaceCoordinate{
      /*node_idx=*/node_idx.raw,
      /*device_idx=*/local_device_id.idx.raw,
    },
    /*device_type=*/local_device_id.device_type,
  };
}

local_device_id_t local_device_id_from_global(global_device_id_t const &global_device_id) {
  return local_device_id_t{
    /*idx=*/device_in_node_idx_t{global_device_id.coord.device_idx},
    /*device_type=*/global_device_id.device_type,
  };
}

} // namespace FlexFlow
