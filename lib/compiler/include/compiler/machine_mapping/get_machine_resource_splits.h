#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_GET_MACHINE_RESOURCE_SPLITS_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_GET_MACHINE_RESOURCE_SPLITS_H

#include "compiler/machine_mapping/machine_compute_resource_slice.dtg.h"
#include <unordered_set>
#include <utility>

namespace FlexFlow {

std::unordered_set<std::pair<MachineComputeResourceSlice, MachineComputeResourceSlice>>
    get_machine_resource_splits(MachineComputeResourceSlice const &resources);

} // namespace FlexFlow

#endif
