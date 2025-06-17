#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CONTAINS_VALUE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CONTAINS_VALUE_H

#include <unordered_map>
#include <map>

namespace FlexFlow {

template <typename K, typename V>
bool contains_value(std::unordered_map<K, V> const &m, V const &v) {
  for (auto const &[kk, vv] : m) {
    if (vv == v) {
      return true;
    }
  }

  return false;
}

template <typename K, typename V>
bool contains_value(std::map<K, V> const &m, V const &v) {
  for (auto const &[kk, vv] : m) {
    if (vv == v) {
      return true;
    }
  }

  return false;
}


} // namespace FlexFlow

#endif
