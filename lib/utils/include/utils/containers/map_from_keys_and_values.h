#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_FROM_KEYS_AND_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_FROM_KEYS_AND_VALUES_H

#include "utils/containers/zip.h"
#include <libassert/assert.hpp>
#include <set>
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V>
std::unordered_map<K, V>
    map_from_keys_and_values(std::vector<K> const &keys,
                             std::vector<V> const &values) {
  ASSERT(keys.size() == values.size());

  std::unordered_map<K, V> result;
  for (auto const &[k, v] : zip(keys, values)) {
    result.insert({k, v});
  }
  return result;
}

} // namespace FlexFlow

#endif
