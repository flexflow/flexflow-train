#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_MACHINE_MAPPING_CACHE_WITH_MEMORY_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_MACHINE_MAPPING_CACHE_WITH_MEMORY_H

#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_cache.dtg.h"

namespace FlexFlow {

MachineMappingWithMemoryCache empty_machine_mapping_with_memory_cache();
std::optional<MachineMappingWithMemoryResult>
    machine_mapping_with_memory_cache_load(
        MachineMappingWithMemoryCache const &, MachineMappingState const &);
void machine_mapping_with_memory_cache_save(
    MachineMappingWithMemoryCache &,
    MachineMappingState const &,
    MachineMappingWithMemoryResult const &);

} // namespace FlexFlow

#endif
