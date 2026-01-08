#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_FILTRANS_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_FILTRANS_VALUES_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename V2 = typename std::invoke_result_t<F, V>::value_type>
bidict<K, V2> filtrans_values(bidict<K, V> const &m, F &&f) {
  bidict<K, V2> result;
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
