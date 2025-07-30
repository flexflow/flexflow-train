#include "pcg/machine_compute_specification.h"
#include "pcg/device_id.h"
#include "utils/containers/transform.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

positive_int get_num_gpus(MachineComputeSpecification const &ms) {
  return ms.num_nodes * ms.num_gpus_per_node;
}

positive_int get_num_cpus(MachineComputeSpecification const &ms) {
  return ms.num_nodes * ms.num_cpus_per_node;
}

positive_int get_num_devices(MachineComputeSpecification const &ms,
                             DeviceType const &device_type) {
  switch (device_type) {
    case DeviceType::GPU:
      return get_num_gpus(ms);
    case DeviceType::CPU:
      return get_num_cpus(ms);
    default:
      PANIC("Unknown DeviceType", device_type);
  }
}

positive_int get_num_devices_per_node(MachineComputeSpecification const &ms,
                                      DeviceType const &device_type) {
  switch (device_type) {
    case DeviceType::GPU:
      return ms.num_gpus_per_node;
    case DeviceType::CPU:
      return ms.num_cpus_per_node;
    default:
      PANIC("Unknown DeviceType", device_type);
  }
}

bool is_valid_machine_space_coordinate(MachineComputeSpecification const &ms,
                                       MachineSpaceCoordinate const &coord) {
  return (coord.node_idx < ms.num_nodes) &&
         (coord.device_idx < get_num_devices_per_node(ms, coord.device_type));
}

device_id_t get_device_id(MachineComputeSpecification const &ms,
                          MachineSpaceCoordinate const &coord) {
  ASSERT(is_valid_machine_space_coordinate(ms, coord));

  nonnegative_int raw_idx =
      coord.node_idx * get_num_devices_per_node(ms, coord.device_type) +
      coord.device_idx;
  return device_id_from_index(raw_idx, coord.device_type);
}

} // namespace FlexFlow
