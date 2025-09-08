#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNIQUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNIQUE_H

#include <unordered_set>
#include <unordered_map>
#include "utils/hash/pair.h"

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
std::unordered_set<T> unordered_set_of(C const &c) {
  return std::unordered_set{c.cbegin(), c.cend()};
}

} // namespace FlexFlow

#endif
