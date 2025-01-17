#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_EXHAUSTIVE_RELATIONAL_JOIN_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_EXHAUSTIVE_RELATIONAL_JOIN_H

#include "utils/bidict/bidict.h"
#include "utils/exception.h"

namespace FlexFlow {

template <typename T1, typename T2, typename T3>
bidict<T1, T3> exhaustive_relational_join(bidict<T1, T2> const &fst, bidict<T2, T3> const &snd) {
  if (fst.size() != snd.size()) {
    throw mk_runtime_error(fmt::format("exhaustive_relational_join received bidicts of different sizes: fst has size {} while snd has size {}", fst.size(), snd.size()));
  }

  bidict<T1, T3> result;

  for (auto const &[v1, v2] : fst) {
    result.equate({v1, snd.at_l(v2)});
  }

  return result;
}

} // namespace FlexFlow

#endif
