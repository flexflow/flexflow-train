#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_UNORDERED_SET_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_UNORDERED_SET_OF_H

#include "utils/hash/pair.h"
#include "utils/unordered_bidict/unordered_bidict.h"

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<std::pair<L, R>>
    unordered_bidict_unordered_set_of(unordered_bidict<L, R> const &c) {
  std::unordered_set<std::pair<L, R>> result;

  for (auto const &lr : c) {
    result.insert(lr);
  }

  return result;
}

} // namespace FlexFlow

#endif
