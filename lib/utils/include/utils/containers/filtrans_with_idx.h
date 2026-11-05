#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTRANS_WITH_IDX_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FILTRANS_WITH_IDX_H

#include "utils/containers/unwrap_optional.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include <vector>

namespace FlexFlow {

template <typename In,
          typename F,
          typename Out =
              unwrap_optional_t<std::invoke_result_t<F, nonnegative_int, In>>>
std::vector<Out> filtrans_with_idx(std::vector<In> const &v, F &&f) {
  std::vector<Out> result;
  for (nonnegative_int idx : nonnegative_range(num_elements(v))) {
    std::optional<Out> out = f(idx, v.at(idx.int_from_nonnegative_int()));
    if (out.has_value()) {
      result.push_back(out.value());
    }
  }
  return result;
}

} // namespace FlexFlow

#endif
