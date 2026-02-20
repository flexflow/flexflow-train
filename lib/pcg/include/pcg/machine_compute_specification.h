#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_COMPUTE_SPECIFICATION_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_COMPUTE_SPECIFICATION_H

#include "pcg/device_id_t.dtg.h"
#include "pcg/device_type.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"

namespace FlexFlow {

positive_int get_num_gpus(MachineComputeSpecification const &ms);
positive_int get_num_cpus(MachineComputeSpecification const &ms);
positive_int get_num_devices(MachineComputeSpecification const &ms,
                             DeviceType const &device_type);
positive_int get_num_devices_per_node(MachineComputeSpecification const &ms,
                                      DeviceType const &device_type);

bool is_valid_machine_space_coordinate(MachineComputeSpecification const &ms,
                                       MachineSpaceCoordinate const &coord);

device_id_t get_device_id(MachineComputeSpecification const &ms,
                          MachineSpaceCoordinate const &coord);

} // namespace FlexFlow

#endif
