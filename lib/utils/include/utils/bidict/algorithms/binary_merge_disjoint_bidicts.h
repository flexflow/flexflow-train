#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BINARY_MERGE_DISJOINT_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BINARY_MERGE_DISJOINT_BIDICTS_H

#include "utils/bidict/bidict.h"
#include <libassert/assert.hpp>
#include "utils/containers/are_disjoint.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/bidict/algorithms/left_entries.h"

namespace FlexFlow {

template <typename K, typename V>
bidict<K, V> binary_merge_disjoint_bidicts(bidict<K, V> const &lhs,
                                           bidict<K, V> const &rhs) {
  ASSERT(
    are_disjoint(left_entries(lhs), left_entries(rhs)),
    "Left entries of {} and {} are non-disjoint", lhs, rhs
  );

  ASSERT(
    are_disjoint(right_entries(lhs), right_entries(rhs)),
    "Right entries of {} and {} are non-disjoint", lhs, rhs
  );

  bidict<K, V> result;
  for (auto const &kv : lhs) {
    result.equate_strict(kv.first, kv.second);
  }
  for (auto const &kv : rhs) {
    result.equate_strict(kv.first, kv.second);
  }

  return result;
}

} // namespace FlexFlow

#endif
