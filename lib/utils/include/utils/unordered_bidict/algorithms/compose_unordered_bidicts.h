#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_COMPOSE_UNORDERED_BIDICTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_COMPOSE_UNORDERED_BIDICTS_H

#include "utils/unordered_bidict/algorithms/unordered_bidict_exhaustive_relational_join.h"

namespace FlexFlow {

template <typename T1, typename T2, typename T3>
unordered_bidict<T1, T3>
    compose_unordered_bidicts(unordered_bidict<T1, T2> const &fst,
                              unordered_bidict<T2, T3> const &snd) {
  return unordered_bidict_exhaustive_relational_join(fst, snd);
}

} // namespace FlexFlow

#endif
