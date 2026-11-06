#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FILTRANS_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FILTRANS_VALUES_H

#include "utils/unordered_bidict/unordered_bidict.h"

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename V2 = typename std::invoke_result_t<F, V>::value_type>
unordered_bidict<K, V2>
    unordered_bidict_filtrans_values(unordered_bidict<K, V> const &m, F &&f) {
  unordered_bidict<K, V2> result;
  for (auto const &[k, v] : m) {
    std::optional<V2> new_v = f(v);
    if (new_v.has_value()) {
      result.equate(k, new_v.value());
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
