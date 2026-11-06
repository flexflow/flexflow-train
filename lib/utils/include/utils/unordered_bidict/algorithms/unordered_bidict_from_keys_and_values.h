#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FROM_KEYS_AND_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FROM_KEYS_AND_VALUES_H

#include "utils/containers/zip.h"
#include "utils/unordered_bidict/algorithms/unordered_bidict_from_pairs.h"
#include "utils/unordered_bidict/unordered_bidict.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename L, typename R>
unordered_bidict<L, R>
    unordered_bidict_from_keys_and_values(std::vector<L> const &ls,
                                          std::vector<R> const &rs) {
  ASSERT(ls.size() == rs.size());

  return unordered_bidict_from_pairs(zip(ls, rs));
}

} // namespace FlexFlow

#endif
