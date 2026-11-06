#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_KEYS_H

#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename K, typename V>
std::set<K> keys(std::map<K, V> const &c) {
  std::set<K> result;
  for (auto const &kv : c) {
    result.insert(kv.first);
  }
  return result;
}

template <typename K, typename V>
std::unordered_set<K> keys(std::unordered_map<K, V> const &c) {
  std::unordered_set<K> result;
  for (auto const &kv : c) {
    result.insert(kv.first);
  }
  return result;
}

} // namespace FlexFlow

#endif
