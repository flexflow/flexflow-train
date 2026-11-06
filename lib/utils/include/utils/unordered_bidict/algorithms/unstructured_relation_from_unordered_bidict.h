#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNSTRUCTURED_RELATION_FROM_UNORDERED_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNSTRUCTURED_RELATION_FROM_UNORDERED_BIDICT_H

#include "utils/unordered_bidict/unordered_bidict.h"
#include <unordered_set>

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<std::pair<L, R>> unstructured_relation_from_unordered_bidict(
    unordered_bidict<L, R> const &b) {

  std::unordered_set<std::pair<L, R>> result;

  for (auto const &lr : b) {
    result.insert(lr);
  }

  return result;
}

} // namespace FlexFlow

#endif
