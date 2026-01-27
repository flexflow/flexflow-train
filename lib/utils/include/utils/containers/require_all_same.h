#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_ALL_SAME_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_ALL_SAME_H

#include "utils/containers/require_all_same1.h"
#include <fmt/format.h>
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
std::optional<T> require_all_same(C const &c) {
  if (c.empty()) {
    return std::nullopt;
  } else {
    return require_all_same1(c);
  }
}

} // namespace FlexFlow

#endif
