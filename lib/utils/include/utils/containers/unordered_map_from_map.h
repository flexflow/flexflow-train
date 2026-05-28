#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNORDERED_MAP_FROM_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNORDERED_MAP_FROM_MAP_H

#include <map>
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V>
std::unordered_map<K, V> unordered_map_from_map(std::map<K, V> const &u) {
  std::unordered_map<K, V> result{u.cbegin(), u.cend()};

  return result;
}

} // namespace FlexFlow

#endif
