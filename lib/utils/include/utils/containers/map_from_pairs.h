#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_FROM_PAIRS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_FROM_PAIRS_H

#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename K, typename V>
std::unordered_map<K, V>
    map_from_pairs(std::unordered_set<std::pair<K, V>> const &pairs) {

  std::unordered_map<K, V> result(pairs.cbegin(), pairs.cend());

  return result;
}

} // namespace FlexFlow

#endif
