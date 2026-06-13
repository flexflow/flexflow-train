#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_DISJOINT_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_MERGE_DISJOINT_BIDICTS_H

#include "utils/bidict/algorithms/binary_merge_disjoint_bidicts.h"
#include "utils/bidict/bidict.h"
#include "utils/containers/foldl.h"

namespace FlexFlow {

template <typename K, typename V>
bidict<K, V> merge_disjoint_bidicts(std::set<bidict<K, V>> const &bidicts) {
  return foldl(
      bidicts,
      bidict<K, V>{},
      [](bidict<K, V> const &accum, bidict<K, V> const &x) -> bidict<K, V> {
        return binary_merge_disjoint_bidicts(accum, x);
      });
}

} // namespace FlexFlow

#endif
