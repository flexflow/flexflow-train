#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPEAT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPEAT_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include <cassert>
#include <type_traits>
#include <vector>

namespace FlexFlow {

template <typename F, typename Out = std::invoke_result_t<F>>
std::vector<Out> repeat(nonnegative_int n, F const &f) {
  std::vector<Out> result;
  for (int i = 0; i < n; i++) {
    result.push_back(f());
  }
  return result;
}

} // namespace FlexFlow

#endif
