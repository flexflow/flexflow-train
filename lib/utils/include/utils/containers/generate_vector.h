#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GENERATE_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GENERATE_VECTOR_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/nonnegative_int/range.h"
#include <type_traits>

namespace FlexFlow {

template <typename F, typename T = std::invoke_result_t<F, nonnegative_int>>
std::vector<T> generate_vector(nonnegative_int length, F &&f) {
  std::vector<T> result;
  for (nonnegative_int idx : range(length)) {
    result.push_back(f(idx));
  }
  return result;
}

} // namespace FlexFlow

#endif
