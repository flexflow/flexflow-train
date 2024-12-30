#include "compiler/machine_mapping/memory_optimization/machine_mapping_with_memory_cache.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/try_at.h"

namespace FlexFlow {

MachineMappingWithMemoryCache empty_machine_mapping_with_memory_cache() {
  return MachineMappingWithMemoryCache{{}};
}

std::optional<MachineMappingWithMemoryResult>
    machine_mapping_with_memory_cache_load(
        MachineMappingWithMemoryCache const &cache,
        MachineMappingState const &k) {
  return try_at(cache.raw_map, k);
}

void machine_mapping_with_memory_cache_save(
    MachineMappingWithMemoryCache &cache,
    MachineMappingState const &k,
    MachineMappingWithMemoryResult const &v) {
  if (contains_key(cache.raw_map, k)) {
    throw mk_runtime_error(fmt::format(
        "machine_mapping_with_memory_cache_save expected key to not already "
        "exist, but received existing key {}",
        k));
  }

  cache.raw_map.emplace(k, v);
}

} // namespace FlexFlow
