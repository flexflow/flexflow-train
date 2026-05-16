#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_DISJOINT_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_DISJOINT_BIDICTS_H

#include "utils/containers/foldl.h"
#include "utils/bidict/algorithms/binary_merge_disjoint_bidicts.h"

namespace FlexFlow {

template <typename C,
          typename K = typename C::value_type::key_type,
          typename V = typename C::value_type::mapped_type>
bidict<K, V> merge_disjoint_bidicts(C const &c) {
  bidict<K, V> empty = {};
  return foldl(c,
               /*init=*/empty,
               [](bidict<K, V> const &lhs,
                  bidict<K, V> const &rhs) {
                 return binary_merge_disjoint_bidicts(lhs, rhs);
               });
}

} // namespace FlexFlow

#endif
