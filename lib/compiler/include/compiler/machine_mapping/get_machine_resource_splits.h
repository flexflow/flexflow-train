#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_GET_MACHINE_RESOURCE_SPLITS_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_GET_MACHINE_RESOURCE_SPLITS_H

#include "pcg/machine_specification.dtg.h"
#include <unordered_set>
#include <utility>

namespace FlexFlow {

std::unordered_set<std::pair<MachineComputeSpecification, MachineComputeSpecification>>
    get_machine_resource_splits(MachineComputeSpecification const &resources);

} // namespace FlexFlow

#endif
