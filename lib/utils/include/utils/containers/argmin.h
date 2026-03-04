#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ARGMIN_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ARGMIN_H

#include "utils/containers/foldl1.h"

namespace FlexFlow {

template <typename C, typename F>
typename C::value_type argmin(C const &c, F &&f) {
  using Elem = typename C::value_type;
  return foldl1(c, [&](Elem const &best, Elem const &candidate) -> Elem {
    if (f(candidate) < f(best)) {
      return candidate;
    }
    return best;
  });
}

} // namespace FlexFlow

#endif
