#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_TRANSFORM_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_TRANSFORM_H

#include "utils/unordered_bidict/unordered_bidict.h"

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename K2 = typename std::invoke_result_t<F, K, V>::first_type,
          typename V2 = typename std::invoke_result_t<F, K, V>::second_type>
unordered_bidict<K2, V2>
    unordered_bidict_transform(unordered_bidict<K, V> const &m, F &&f) {
  unordered_bidict<K2, V2> result;
  for (auto const &[k, v] : m) {
    result.equate(f(k, v));
  }
  return result;
}

} // namespace FlexFlow

#endif
