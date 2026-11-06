#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VALUES_H

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename K, typename V>
std::multiset<V> values(std::map<K, V> const &c) {
  std::multiset<V> result;
  for (auto const &kv : c) {
    result.insert(kv.second);
  }
  return result;
}

template <typename K, typename V>
std::unordered_multiset<V> values(std::unordered_map<K, V> const &c) {
  std::unordered_multiset<V> result;
  for (auto const &kv : c) {
    result.insert(kv.second);
  }
  return result;
}

} // namespace FlexFlow

#endif
