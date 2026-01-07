#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_ALL_SAME1_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_ALL_SAME1_H

#include <fmt/format.h>
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
T require_all_same1(C const &c) {
  ASSERT(!c.empty());

  T const &first = *c.cbegin();
  for (T const &v : c) {
    ASSERT(v == first);
  }
  return first;
}

} // namespace FlexFlow

#endif
