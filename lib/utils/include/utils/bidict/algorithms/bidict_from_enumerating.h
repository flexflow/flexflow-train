#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_ENUMERATING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_BIDICT_FROM_ENUMERATING_H

#include "utils/bidict/bidict.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <set>

namespace FlexFlow {

template <typename T>
bidict<nonnegative_int, T>
    bidict_from_enumerating(std::unordered_set<T> const &s) {
  bidict<nonnegative_int, T> result;
  nonnegative_int idx = 0_n;
  for (T const &t : s) {
    result.equate(idx, t);
    idx++;
  }

  return result;
}

template <typename T>
bidict<nonnegative_int, T> bidict_from_enumerating(std::set<T> const &s) {
  bidict<nonnegative_int, T> result;
  nonnegative_int idx = 0_n;
  for (T const &t : s) {
    result.equate(idx, t);
    idx++;
  }

  return result;
}

} // namespace FlexFlow

#endif
