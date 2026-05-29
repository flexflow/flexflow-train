#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_DISJOINT_MAPS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_DISJOINT_MAPS_H

#include "utils/containers/binary_merge_maps_with.h"
#include <libassert/assert.hpp>
#include "utils/containers/keys.h"
#include "utils/containers/intersection.h"

namespace FlexFlow {

template <typename K, typename V>
std::map<K, V>
    binary_merge_disjoint_maps(std::map<K, V> const &lhs,
                               std::map<K, V> const &rhs) {

  std::set<K> lhs_keys = keys(lhs);
  std::set<K> rhs_keys = keys(rhs);

  std::set<K> shared_keys = intersection(lhs_keys, rhs_keys);
  ASSERT(shared_keys.empty());

  return binary_merge_maps_with(
      lhs, rhs, [](V const &, V const &) -> V { PANIC(); });
}

} // namespace FlexFlow

#endif
