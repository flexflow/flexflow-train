#ifndef _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_MACHINE_MAPPING_CACHE_WITH_MEMORY_H
#define _FLEXFLOW_LIB_COMPILER_INCLUDE_COMPILER_MACHINE_MAPPING_MEMORY_OPTIMIZATION_MACHINE_MAPPING_CACHE_WITH_MEMORY_H

#include "compiler/machine_mapping/memory_optimization/machine_mapping_cache_with_memory.dtg.h"

namespace FlexFlow {

MachineMappingCacheWithMemory empty_machine_mapping_cache_with_memory();
std::optional<MachineMappingResultWithMemory>
    machine_mapping_cache_with_memory_load(
        MachineMappingCacheWithMemory const &, MachineMappingState const &);
void machine_mapping_cache_with_memory_save(
    MachineMappingCacheWithMemory &,
    MachineMappingState const &,
    MachineMappingResultWithMemory const &);

} // namespace FlexFlow

#endif
