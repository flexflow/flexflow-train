#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ARE_DISJOINT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ARE_DISJOINT_H

#include "utils/containers/set_intersection.h"

namespace FlexFlow {

template <typename T>
bool are_disjoint(std::unordered_set<T> const &l,
                  std::unordered_set<T> const &r) {
  return set_intersection<T>(l, r).empty();
}

} // namespace FlexFlow

#endif
