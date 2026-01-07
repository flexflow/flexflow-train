#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_COUNT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_COUNT_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include <cstddef>
#include <vector>

namespace FlexFlow {

template <typename C, typename F>
nonnegative_int count(C const &c, F const &f) {
  nonnegative_int result = 0_n;
  for (auto const &v : c) {
    if (f(v)) {
      result++;
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
