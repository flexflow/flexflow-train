#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MULTISET_OF_H

#include <set>

namespace FlexFlow {

template <typename C, typename T = typename C::value_type>
std::multiset<T> multiset_of(C const &c) {
  return std::multiset<T>{std::cbegin(c), std::cend(c)};
}

} // namespace FlexFlow

#endif
