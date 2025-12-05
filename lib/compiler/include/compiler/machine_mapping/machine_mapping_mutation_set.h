#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MCMC_MACHINE_MAPPING_MUTATION_SET_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MCMC_MACHINE_MAPPING_MUTATION_SET_H

#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/search_result.dtg.h"

namespace FlexFlow {
std::optional<MachineMapping>
    get_random_mapping(ParallelComputationGraph &pcg,
                       MachineSpecification const &resources,
                       DeviceType const &device_type);

std::optional<MachineMapping>
    get_random_mutation(SearchResult mapped_pcg,
                        MachineSpecification const &resource,
                        DeviceType const &device_type);
} // namespace FlexFlow

#endif
