#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_UNORDERED_SET_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_UNORDERED_SET_OF_H

#include "utils/hash/pair.h"
#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<std::pair<L, R>> unordered_set_of(bidict<L, R> const &c) {
  std::unordered_set<std::pair<L, R>> result;

  for (auto const &lr : c) {
    result.insert(lr);
  }

  return result;
}


} // namespace FlexFlow

#endif
