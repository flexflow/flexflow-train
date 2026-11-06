#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_RIGHT_ENTRIES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_RIGHT_ENTRIES_H

#include "utils/unordered_bidict/unordered_bidict.h"
#include <unordered_set>

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<R>
    unordered_bidict_right_entries(unordered_bidict<L, R> const &b) {
  return b.right_values();
}

} // namespace FlexFlow

#endif
