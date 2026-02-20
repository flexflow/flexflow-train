#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_MAPS_WITH_RIGHT_DOMINATING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_MAPS_WITH_RIGHT_DOMINATING_H

#include "utils/containers/merge_in_map.h"

namespace FlexFlow {

template <typename C,
          typename K = typename C::value_type::key_type,
          typename V = typename C::value_type::mapped_type>
std::unordered_map<K, V> merge_maps_with_right_dominating(C const &c) {
  std::unordered_map<K, V> result;

  for (std::unordered_map<K, V> const &m : c) {
    merge_in_map(m, result);
  }

  return result;
}

} // namespace FlexFlow

#endif
