#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_EXHAUSTIVE_RELATIONAL_JOIN_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_EXHAUSTIVE_RELATIONAL_JOIN_H

#include "utils/bidict/algorithms/bidict_left_entries.h"
#include "utils/bidict/algorithms/bidict_right_entries.h"
#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename T1, typename T2, typename T3>
bidict<T1, T3> bidict_exhaustive_relational_join(bidict<T1, T2> const &fst,
                                                 bidict<T2, T3> const &snd) {
  ASSERT(bidict_right_entries(fst) == bidict_left_entries(snd));

  bidict<T1, T3> result;

  for (auto const &[v1, v2] : fst) {
    result.equate({v1, snd.at_l(v2)});
  }

  return result;
}

} // namespace FlexFlow

#endif
