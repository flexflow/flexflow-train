#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_FILTER_UNORDERED_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_UNORDERED_BIDICT_ALGORITHMS_FILTER_UNORDERED_BIDICT_H

#include "utils/unordered_bidict/unordered_bidict.h"

namespace FlexFlow {

template <typename L, typename R, typename F>
unordered_bidict<L, R> filter_unordered_bidict(unordered_bidict<L, R> const &b, F &&f) {
  unordered_bidict<L, R> result;

  for (std::pair<L, R> const &p : b) {
    if (f(p.first, p.second)) {
      result.equate(p.first, p.second);
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
