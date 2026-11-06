#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_BINARY_MERGE_DISJOINT_UNORDERED_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_BINARY_MERGE_DISJOINT_UNORDERED_BIDICTS_H

#include "utils/unordered_bidict/algorithms/left_entries.h"
#include "utils/unordered_bidict/algorithms/right_entries.h"
#include "utils/unordered_bidict/unordered_bidict.h"
#include "utils/containers/are_disjoint.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename K, typename V>
unordered_bidict<K, V> binary_merge_disjoint_unordered_bidicts(unordered_bidict<K, V> const &lhs,
                                           unordered_bidict<K, V> const &rhs) {
  ASSERT(are_disjoint(left_entries(lhs), left_entries(rhs)),
         "Left entries of {} and {} are non-disjoint",
         lhs,
         rhs);

  ASSERT(are_disjoint(right_entries(lhs), right_entries(rhs)),
         "Right entries of {} and {} are non-disjoint",
         lhs,
         rhs);

  unordered_bidict<K, V> result;
  for (auto const &kv : lhs) {
    result.equate(kv.first, kv.second);
  }
  for (auto const &kv : rhs) {
    result.equate(kv.first, kv.second);
  }

  return result;
}

} // namespace FlexFlow

#endif
