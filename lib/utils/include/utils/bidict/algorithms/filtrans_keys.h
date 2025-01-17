#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_FILTRANS_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_FILTRANS_KEYS_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename K2 = typename std::invoke_result_t<F, K>::value_type>
bidict<K2, V> filtrans_keys(bidict<K, V> const &m, F &&f) {
  bidict<K2, V> result;
  for (auto const &[k, v] : m) {
    std::optional<K2> new_k = f(k);
    if (new_k.has_value()) {
      result.equate(new_k.value(), v);
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
