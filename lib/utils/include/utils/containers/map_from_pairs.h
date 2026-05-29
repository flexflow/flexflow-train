#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_FROM_PAIRS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_FROM_PAIRS_H

#include <map>

namespace FlexFlow {

template <typename C,
          typename K = typename C::value_type::first_type,
          typename V = typename C::value_type::second_type>
std::map<K, V> map_from_pairs(C const &c) {
  return std::map<K, V>(c.cbegin(), c.cend());
}

} // namespace FlexFlow

#endif
