#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_DISJOINT_MAPS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_DISJOINT_MAPS_H

#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/foldl.h"

namespace FlexFlow {

template <typename C, 
          typename K = typename C::value_type::key_type,
          typename V = typename C::value_type::mapped_type>
std::unordered_map<K, V> merge_disjoint_maps(C const &c) {
  std::unordered_map<K, V> empty = {};
  return foldl(
    c, 
    /*init=*/empty, 
    [](std::unordered_map<K, V> const &lhs, 
       std::unordered_map<K, V> const &rhs) {
      return binary_merge_disjoint_maps(lhs, rhs); 
    });
}


} // namespace FlexFlow

#endif
