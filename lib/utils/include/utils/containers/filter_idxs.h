#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTER_IDXS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTER_IDXS_H

#include "utils/nonnegative_int/nonnegative_int.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/nonnegative_int/range.h"
#include <functional>
#include <vector>

namespace FlexFlow {

template <typename T>
std::vector<T> filter_idxs(std::vector<T> const &input,
                           std::function<bool(nonnegative_int)> const &f) {
  std::vector<T> result;

  for (nonnegative_int idx : range(num_elements(input))) {
    if (f(idx)) {
      result.push_back(input.at(idx.unwrap_nonnegative()));
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
