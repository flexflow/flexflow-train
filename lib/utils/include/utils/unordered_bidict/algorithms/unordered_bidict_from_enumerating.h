#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FROM_ENUMERATING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_UNORDERED_BIDICT_FROM_ENUMERATING_H

#include "utils/unordered_bidict/unordered_bidict.h"
#include "utils/containers/contains_duplicates.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <libassert/assert.hpp>
#include <set>

namespace FlexFlow {

template <typename T>
unordered_bidict<nonnegative_int, T> unordered_bidict_from_enumerating(std::vector<T> const &s) {
  ASSERT(!contains_duplicates(s));

  unordered_bidict<nonnegative_int, T> result;
  nonnegative_int idx = 0_n;
  for (T const &t : s) {
    result.equate(idx, t);
    idx++;
  }

  return result;
}

template <typename T>
unordered_bidict<nonnegative_int, T>
    unordered_bidict_from_enumerating(std::unordered_set<T> const &s) {
  unordered_bidict<nonnegative_int, T> result;
  nonnegative_int idx = 0_n;
  for (T const &t : s) {
    result.equate(idx, t);
    idx++;
  }

  return result;
}

template <typename T>
unordered_bidict<nonnegative_int, T> unordered_bidict_from_enumerating(std::set<T> const &s) {
  unordered_bidict<nonnegative_int, T> result;
  nonnegative_int idx = 0_n;
  for (T const &t : s) {
    result.equate(idx, t);
    idx++;
  }

  return result;
}

} // namespace FlexFlow

#endif
