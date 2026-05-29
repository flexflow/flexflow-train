#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_UNORDERED_MAPS_WITH_LEFT_DOMINATING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_UNORDERED_MAPS_WITH_LEFT_DOMINATING_H

#include "utils/containers/merge_in_unordered_map.h"

namespace FlexFlow {

template <typename K, typename V>
std::unordered_map<K, V> binary_merge_unordered_maps_with_left_dominating(
    std::unordered_map<K, V> const &lhs, std::unordered_map<K, V> const &rhs) {
  std::unordered_map<K, V> result;
  merge_in_unordered_map(rhs, result);
  merge_in_unordered_map(lhs, result);
  return result;
}

} // namespace FlexFlow

#endif
