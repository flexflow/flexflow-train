#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_UNSTRUCTURED_RELATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_UNSTRUCTURED_RELATION_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename L, typename R>
bidict<L, R> bidict_from_unstructured_relation(
    std::unordered_set<std::pair<L, R>> const &relation) {
  bidict<L, R> result;
  for (auto const &lr : relation) {
    result.equate_strict(lr);
  }
  return result;
}

} // namespace FlexFlow

#endif
