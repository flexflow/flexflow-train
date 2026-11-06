#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_TRANSFORM_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_TRANSFORM_KEYS_H

#include "utils/unordered_bidict/unordered_bidict.h"

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename K2 = std::invoke_result_t<F, K>>
unordered_bidict<K2, V> unordered_bidict_transform_keys(unordered_bidict<K, V> const &m, F &&f) {
  unordered_bidict<K2, V> result;
  for (auto const &kv : m) {
    result.equate(f(kv.first), kv.second);
  }
  return result;
}

} // namespace FlexFlow

#endif
