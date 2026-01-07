#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_PARETO_OPTIMAL_MACHINE_MAPPING_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_PARETO_OPTIMAL_MACHINE_MAPPING_H

#include "compiler/machine_mapping/memory_optimization/pareto_optimal_machine_mapping.dtg.h"

namespace FlexFlow {

bool is_pareto_optimal_in(
    ParetoOptimalMachineMapping const &,
    std::unordered_set<ParetoOptimalMachineMapping> const &);

} // namespace FlexFlow

#endif
