#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_TRANSFORM_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_TRANSFORM_KEYS_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename K2 = std::invoke_result_t<F, K>>
bidict<K2, V> transform_keys(bidict<K, V> const &m, F &&f) {
  bidict<K2, V> result;
  for (auto const &kv : m) {
    result.equate(f(kv.first), kv.second);
  }
  return result;
}

} // namespace FlexFlow

#endif
