#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_COMPUTE_RESOURCE_SLICE_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MACHINE_COMPUTE_RESOURCE_SLICE_H

#include "pcg/machine_compute_specification.dtg.h"
#include "compiler/machine_mapping/machine_compute_resource_slice.dtg.h"

namespace FlexFlow {

MachineComputeResourceSlice
  compute_slice_from_specification(MachineComputeSpecification const &);

} // namespace FlexFlow

#endif
