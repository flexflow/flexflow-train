#ifndef _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PCG_ARGS_BACKING_H
#define _FLEXFLOW_LIB_LOCAL_PCG_EXECUTION_INCLUDE_LOCAL_PCG_EXECUTION_LOCAL_PCG_ARGS_BACKING_H

#include "local-pcg-execution/local_pcg_args_backing.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"
#include "task-spec/device_specific_per_device_op_state.dtg.h"
#include "task-spec/symbolic_layer_guid_t.dtg.h"
#include <optional>
#include <unordered_map>

namespace FlexFlow {

std::unordered_map<symbolic_layer_guid_t, std::optional<DeviceSpecificPerDeviceOpState>>
  get_op_states_for_machine_space_coord(LocalPcgArgsBacking const &, MachineSpaceCoordinate const &);

} // namespace FlexFlow

#endif
