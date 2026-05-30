#include "pcg/machine_compute_specification.h"
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

} // namespace FlexFlow
