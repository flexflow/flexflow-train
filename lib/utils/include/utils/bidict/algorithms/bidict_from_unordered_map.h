#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_UNORDERED_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_UNORDERED_MAP_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename L, typename R>
bidict<L, R> bidict_from_unordered_map(std::unordered_map<L, R> const &m) {
  return bidict<L, R>{m.begin(), m.end()};
}

} // namespace FlexFlow

#endif
