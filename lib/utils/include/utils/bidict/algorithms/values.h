#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_VALUES_H

#include "utils/bidict/bidict.h"
#include <map>
#include <set>

namespace FlexFlow {

template <typename K, typename V>
std::set<V> values(bidict<K, V> const &c) {
  return c.right_values();
}

} // namespace FlexFlow

#endif
