#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_TRANSFORM_KEYS_AND_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_TRANSFORM_KEYS_AND_VALUES_H

#include "utils/unordered_bidict/unordered_bidict.h"

namespace FlexFlow {

template <typename K,
          typename V,
          typename KF,
          typename VF,
          typename K2 = std::invoke_result_t<KF, K>,
          typename V2 = std::invoke_result_t<VF, V>>
unordered_bidict<K2, V2> unordered_bidict_transform_keys_and_values(
    unordered_bidict<K, V> const &m, KF &&kf, VF &&vf) {
  unordered_bidict<K2, V2> result;
  for (auto const &kv : m) {
    result.equate(kf(kv.first), vf(kv.second));
  }
  return result;
}

} // namespace FlexFlow

#endif
