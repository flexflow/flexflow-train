#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REPLICATE_H

#include "utils/exception.h"
#include <fmt/format.h>
#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<T> replicate(int n, T const &element) {
  if (n < 0) {
    throw mk_runtime_error(
        fmt::format("replicate expected n > 0, but received n = {}", n));
  }

  std::vector<T> result;
  for (int i = 0; i < n; ++i) {
    result.push_back(element);
  }
  return result;
}

} // namespace FlexFlow

#endif
