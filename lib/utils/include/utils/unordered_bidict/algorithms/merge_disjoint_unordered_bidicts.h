#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_MERGE_DISJOINT_UNORDERED_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_MERGE_DISJOINT_UNORDERED_BIDICTS_H

#include "utils/unordered_bidict/algorithms/binary_merge_disjoint_unordered_bidicts.h"
#include "utils/unordered_bidict/unordered_bidict.h"
#include "utils/containers/foldl.h"

namespace FlexFlow {

template <typename C,
          typename K = typename C::value_type::key_type,
          typename V = typename C::value_type::mapped_type>
unordered_bidict<K, V> merge_disjoint_unordered_bidicts(C const &c) {
  unordered_bidict<K, V> empty = {};
  return foldl(c,
               /*init=*/empty,
               [](unordered_bidict<K, V> const &lhs, unordered_bidict<K, V> const &rhs) {
                 return binary_merge_disjoint_unordered_bidicts(lhs, rhs);
               });
}

} // namespace FlexFlow

#endif
