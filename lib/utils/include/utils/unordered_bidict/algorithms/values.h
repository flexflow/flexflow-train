#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_VALUES_H

#include "utils/unordered_bidict/unordered_bidict.h"
#include <unordered_map>
#include <unordered_set>

namespace FlexFlow {

template <typename K, typename V>
std::unordered_set<V> values(unordered_bidict<K, V> const &c) {
  return c.right_values();
}

} // namespace FlexFlow

#endif
