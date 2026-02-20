#include "local-pcg-execution/local_pcg_args_backing.h"

namespace FlexFlow {

std::unordered_map<symbolic_layer_guid_t,
                   std::optional<DeviceSpecificPerDeviceOpState>>
    get_op_states_for_machine_space_coord(
        LocalPcgArgsBacking const &args_backing,
        MachineSpaceCoordinate const &coord) {

  return map_values(
      args_backing.per_device_op_states,
      [&](std::optional<MappedPerDeviceOpStatesGroup> const &m_g) {
        return transform(m_g, [&](MappedPerDeviceOpStatesGroup const &g) {
          return g.get_per_device_op_states().at_l(coord);
        });
      });
}

} // namespace FlexFlow
