#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_TRANSFORM_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_TRANSFORM_VALUES_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename K,
          typename V,
          typename F,
          typename V2 = std::invoke_result_t<F, V>>
bidict<K, V2> transform_values(bidict<K, V> const &m, F &&f) {
  bidict<K, V2> result;
  for (auto const &kv : m) {
    result.equate({kv.first, f(kv.second)});
  }
  return result;
}

} // namespace FlexFlow

#endif
