#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_MAPS_WITH_RIGHT_DOMINATING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_MAPS_WITH_RIGHT_DOMINATING_H

#include "utils/containers/merge_in_map.h"

namespace FlexFlow {

template <typename K, typename V>
std::unordered_map<K, V> binary_merge_maps_with_right_dominating(
    std::unordered_map<K, V> const &lhs, std::unordered_map<K, V> const &rhs) {
  std::unordered_map<K, V> result;
  merge_in_map(lhs, result);
  merge_in_map(rhs, result);
  return result;
}

} // namespace FlexFlow

#endif
