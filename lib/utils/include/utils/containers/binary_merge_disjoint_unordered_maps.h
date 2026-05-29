#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_DISJOINT_UNORDERED_MAPS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_DISJOINT_UNORDERED_MAPS_H

#include <libassert/assert.hpp>
#include "utils/containers/binary_merge_unordered_maps_with.h"
#include "utils/containers/unordered_keys.h"
#include "utils/containers/intersection.h"

namespace FlexFlow {

template <typename K, typename V>
std::unordered_map<K, V>
    binary_merge_disjoint_unordered_maps(std::unordered_map<K, V> const &lhs,
                               std::unordered_map<K, V> const &rhs) {

  std::unordered_set<K> lhs_keys = unordered_keys(lhs);
  std::unordered_set<K> rhs_keys = unordered_keys(rhs);

  std::unordered_set<K> shared_keys = intersection(lhs_keys, rhs_keys);
  ASSERT(shared_keys.empty());

  return binary_merge_unordered_maps_with(
      lhs, rhs, [](V const &, V const &) -> V { PANIC(); });
}

} // namespace FlexFlow

#endif
