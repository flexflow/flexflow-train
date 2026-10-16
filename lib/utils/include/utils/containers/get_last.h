#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_LAST_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_LAST_H

#include <libassert/assert.hpp>
#include <vector>

namespace FlexFlow {

template <typename T>
T const &get_last(std::vector<T> const &v) {
  ASSERT(!v.empty());
  return v.back();
}

} // namespace FlexFlow

#endif
