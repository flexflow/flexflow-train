#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_DISJOINT_SETS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_DISJOINT_SETS_H

#include "utils/containers/binary_merge_disjoint_sets.h"
#include "utils/containers/foldl.h"
#include "utils/containers/get_element_type.h"
#include <set>

namespace FlexFlow {

template <typename C, typename T = get_element_type_t<get_element_type_t<C>>>
std::set<T> merge_disjoint_sets(C const &c) {
  std::set<T> empty = {};
  return foldl(
      c,
      /*init=*/empty,
      [](std::set<T> const &lhs, std::set<T> const &rhs) -> std::set<T> {
        return binary_merge_disjoint_sets(lhs, rhs);
      });
}

} // namespace FlexFlow

#endif
