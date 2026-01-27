#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_SAME_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_SAME_H

#include "utils/exception.h"
#include <fmt/format.h>

namespace FlexFlow {

template <typename T>
T const &require_same(T const &l, T const &r) {
  ASSERT(l == r, "require_same received non-equal inputs");

  return l;
}

template <typename T>
T const &require_same(T const &t1, T const &t2, T const &t3) {
  return require_same(require_same(t1, t2), t3);
}

template <typename T>
T const &require_same(T const &t1, T const &t2, T const &t3, T const &t4) {
  return require_same(require_same(require_same(t1, t2), t3), t4);
}

} // namespace FlexFlow

#endif
