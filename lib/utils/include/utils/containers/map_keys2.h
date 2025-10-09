#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_KEYS2_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_KEYS2_H

#include <unordered_map>

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename K2 = std::invoke_result_t<F, K, V>>
std::unordered_map<K2, V> map_keys(std::unordered_map<K, V> const &m,
                                   F const &f) {

  std::unordered_map<K2, V> result;
  for (auto const &kv : m) {
    result.insert({f(kv.first, kv.second), kv.second});
  }

  ASSERT(keys(m).size() == keys(result).size(), 
         "keys passed to map_keys must be transformed into distinct keys");

  return result;
}


} // namespace FlexFlow

#endif
