#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FIND_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FIND_H

#include <algorithm>
#include <unordered_set>

namespace FlexFlow {

template <typename Container>
typename Container::const_iterator
    find(Container const &c, typename Container::value_type const &e) {
  return std::find(c.cbegin(), c.cend(), e);
}

template <typename V>
typename std::unordered_set<V>::const_iterator
    find(std::unordered_set<V> const &c, V const &e) {
  return c.find(e);
}

} // namespace FlexFlow

#endif
