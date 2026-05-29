#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_FROM_UNORDERED_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_FROM_UNORDERED_H

#include <map>
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V>
std::map<K, V> map_from_unordered(std::unordered_map<K, V> const &u) {
  std::map<K, V> result{u.cbegin(), u.cend()};

  return result;
}

} // namespace FlexFlow

#endif
