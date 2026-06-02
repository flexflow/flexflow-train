#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_FILTER_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_FILTER_BIDICT_H

#include "utils/bidict/bidict.h"

namespace FlexFlow {

template <typename L, typename R, typename F>
bidict<L, R> filter_bidict(bidict<L, R> const &b, F &&f) {
  bidict<L, R> result;

  for (std::pair<L, R> const &p : b) {
    if (f(p.first, p.second)) {
      result.equate_strict(p.first, p.second);
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
