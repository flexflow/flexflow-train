#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_UNORDERED_SET_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_UNORDERED_SET_OF_H

#include "utils/hash/pair.h"
#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename K, typename V>
std::unordered_set<std::pair<K, V>> unordered_set_of(bidict<K, V> const &c) {
  return std::unordered_set{c.cbegin(), c.cend()};
}


} // namespace FlexFlow

#endif
