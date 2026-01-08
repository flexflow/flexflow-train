#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_MAPS_WITH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_MAPS_WITH_H

#include "utils/containers/binary_merge_maps_with.h"
#include "utils/containers/foldl.h"
#include <unordered_map>
#include <vector>

namespace FlexFlow {

template <typename K, typename V, typename F>
std::unordered_map<K, V>
    merge_maps_with(std::vector<std::unordered_map<K, V>> const &to_merge,
                    F &&f) {
  return foldl(to_merge,
               std::unordered_map<K, V>{},
               [&](std::unordered_map<K, V> const &accum,
                   std::unordered_map<K, V> const &m) {
                 return binary_merge_maps_with(accum, m, f);
               });
}

} // namespace FlexFlow

#endif
