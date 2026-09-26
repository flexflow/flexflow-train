#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_DISJOINT_SETS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_MERGE_DISJOINT_SETS_H

#include "utils/containers/are_disjoint.h"
#include "utils/containers/set_union.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename T>
std::set<T> binary_merge_disjoint_sets(std::set<T> const &lhs,
                                       std::set<T> const &rhs) {
  ASSERT(are_disjoint(lhs, rhs));

  return set_union(lhs, rhs);
}

} // namespace FlexFlow

#endif
