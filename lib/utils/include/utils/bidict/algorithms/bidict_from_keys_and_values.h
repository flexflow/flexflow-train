#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_KEYS_AND_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_KEYS_AND_VALUES_H

#include "utils/bidict/algorithms/bidict_from_pairs.h"
#include "utils/bidict/bidict.h"
#include "utils/containers/zip.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename L, typename R>
bidict<L, R> bidict_from_keys_and_values(std::vector<L> const &ls,
                                         std::vector<R> const &rs) {
  ASSERT(ls.size() == rs.size());

  return bidict_from_pairs(zip(ls, rs));
}

} // namespace FlexFlow

#endif
