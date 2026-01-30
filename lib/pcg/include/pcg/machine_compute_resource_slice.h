#ifndef _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_COMPUTE_RESOURCE_SLICE_H
#define _FLEXFLOW_LIB_PCG_INCLUDE_PCG_MACHINE_COMPUTE_RESOURCE_SLICE_H

#include "pcg/machine_compute_resource_slice.dtg.h"
#include "pcg/machine_compute_specification.dtg.h"
#include "pcg/machine_space_coordinate.dtg.h"

namespace FlexFlow {

MachineComputeResourceSlice
    compute_slice_from_specification(MachineComputeSpecification const &);

positive_int
    get_total_num_devices_in_slice(MachineComputeResourceSlice const &);

bool is_valid_machine_space_coordinate_in_slice(
    MachineComputeResourceSlice const &slice,
    MachineSpaceCoordinate const &coord);

} // namespace FlexFlow

#endif
