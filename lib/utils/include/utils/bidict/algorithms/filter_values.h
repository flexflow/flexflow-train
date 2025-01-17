#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_FILTER_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_FILTER_VALUES_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename K, typename V, typename F>
bidict<K, V> filter_values(bidict<K, V> const &m, F &&f) {
  bidict<K, V> result;
  for (auto const &kv : m) {
    if (f(kv.second)) {
      result.equate(kv);
    }
  }
  return result;
}


} // namespace FlexFlow

#endif
