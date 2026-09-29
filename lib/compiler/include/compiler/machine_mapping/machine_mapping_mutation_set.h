#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MCMC_MACHINE_MAPPING_MUTATION_SET_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MCMC_MACHINE_MAPPING_MUTATION_SET_H

#include "compiler/machine_mapping/machine_mapping.h"
#include "compiler/search_result.dtg.h"
#include <random>

namespace FlexFlow {
std::optional<MachineMapping>
    get_random_mapping(std::mt19937 &gen,
                       ParallelComputationGraph const &pcg,
                       MachineComputeSpecification const &resources);

std::optional<MachineMapping>
    get_random_mutation(std::mt19937 &gen,
                        SearchResult const &mapped_pcg,
                        MachineComputeSpecification const &resource);
} // namespace FlexFlow

#endif
