#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_VALUES2_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_VALUES2_H

#include <type_traits>
#include <unordered_map>

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename V2 = std::invoke_result_t<F, K, V>>
std::unordered_map<K, V2> map_values2(std::unordered_map<K, V> const &m,
                                      F &&f) {
  std::unordered_map<K, V2> result;
  for (std::pair<K, V> const &kv : m) {
    result.insert(std::pair<K, V2>{kv.first, f(kv.first, kv.second)});
  }
  return result;
}

} // namespace FlexFlow

#endif
