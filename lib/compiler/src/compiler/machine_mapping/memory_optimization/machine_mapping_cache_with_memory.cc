#include "compiler/machine_mapping/memory_optimization/machine_mapping_cache_with_memory.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/try_at.h"

namespace FlexFlow {

MachineMappingCacheWithMemory empty_machine_mapping_cache_with_memory() {
  return MachineMappingCacheWithMemory{{}};
}

std::optional<MachineMappingResultWithMemory>
    machine_mapping_cache_with_memory_load(
        MachineMappingCacheWithMemory const &cache,
        MachineMappingState const &k) {
  return try_at(cache.raw_map, k);
}

void machine_mapping_cache_with_memory_save(
    MachineMappingCacheWithMemory &cache,
    MachineMappingState const &k,
    MachineMappingResultWithMemory const &v) {
  if (contains_key(cache.raw_map, k)) {
    throw mk_runtime_error(fmt::format(
        "machine_mapping_cache_with_memory_save expected key to not already "
        "exist, but received existing key {}",
        k));
  }

  cache.raw_map.emplace(k, v);
}

} // namespace FlexFlow
