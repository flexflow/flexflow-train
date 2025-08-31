#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_KEYS_WITH_VALUE_MERGING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MAP_KEYS_WITH_VALUE_MERGING_H

#include <unordered_map>
#include "utils/containers/contains_key.h"

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename MergeF,
          typename K2 = std::invoke_result_t<F, K>>
std::unordered_map<K2, V> map_keys_with_value_merging(std::unordered_map<K, V> const &m,
                                   F &&key_func,
                                   MergeF &&merge_values) {

  std::unordered_map<K2, V> result;

  for (auto const &kv : m) {
    K k = kv.first;
    V v = kv.second;

    K2 k2 = key_func(k);

    if (contains_key(result, k2)) {
      result.at(k2) = merge_values(result.at(k2), v);
    } else {
      result.insert({k2, v});
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
