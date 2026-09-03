#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_IS_EMPTY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_IS_EMPTY_H

#include <iterator>

namespace FlexFlow {

template <typename C>
bool is_empty(C const &c) {
  return std::begin(c) == std::end(c);
}

} // namespace FlexFlow

#endif
