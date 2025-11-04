#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAXIMUM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAXIMUM_H

#include <libassert/assert.hpp>
#include <algorithm>
#include <fmt/format.h>

namespace FlexFlow {

template <typename C>
typename C::value_type maximum(C const &c) {
  if (c.empty()) {
    PANIC(
        fmt::format("maximum expected non-empty container but received {}", c));
  }

  return *std::max_element(c.begin(), c.end());
}

} // namespace FlexFlow

#endif
