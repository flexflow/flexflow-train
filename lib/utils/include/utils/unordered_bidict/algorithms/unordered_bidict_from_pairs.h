#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FROM_PAIRS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FROM_PAIRS_H

#include "utils/unordered_bidict/unordered_bidict.h"

namespace FlexFlow {

template <typename C,
          typename L = typename C::value_type::first_type,
          typename R = typename C::value_type::second_type>
unordered_bidict<L, R> unordered_bidict_from_pairs(C const &c) {
  return unordered_bidict<L, R>{c.begin(), c.end()};
}

} // namespace FlexFlow

#endif
