#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ARGMAX_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ARGMAX_H

#include "utils/containers/foldl1.h"

namespace FlexFlow {

template <typename C, typename F>
typename C::value_type argmax(C const &c, F &&f) {
  using Elem = typename C::value_type;
  return foldl1(c, [&](Elem const &best, Elem const &candidate) -> Elem {
    if (f(best) < f(candidate)) {
      return candidate;
    }
    return best;
  });
}

} // namespace FlexFlow

#endif
