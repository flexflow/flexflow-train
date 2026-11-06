#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_EXHAUSTIVE_RELATIONAL_JOIN_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_EXHAUSTIVE_RELATIONAL_JOIN_H

#include "utils/unordered_bidict/algorithms/unordered_bidict_left_entries.h"
#include "utils/unordered_bidict/algorithms/unordered_bidict_right_entries.h"
#include "utils/unordered_bidict/unordered_bidict.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename T1, typename T2, typename T3>
unordered_bidict<T1, T3> unordered_bidict_exhaustive_relational_join(
    unordered_bidict<T1, T2> const &fst, unordered_bidict<T2, T3> const &snd) {
  ASSERT(unordered_bidict_right_entries(fst) ==
         unordered_bidict_left_entries(snd));

  unordered_bidict<T1, T3> result;

  for (auto const &[v1, v2] : fst) {
    result.equate({v1, snd.at_l(v2)});
  }

  return result;
}

} // namespace FlexFlow

#endif
