#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_WITH_STRICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_WITH_STRICT_H

#include "utils/containers/zip_with.h"
#include "utils/exception.h"
#include "utils/fmt/vector.h"
#include <vector>

namespace FlexFlow {

template <typename T1,
          typename T2,
          typename F,
          typename Result = std::invoke_result_t<F, T1, T2>>
std::vector<Result> zip_with_strict(std::vector<T1> const &lhs,
                                    std::vector<T2> const &rhs,
                                    F &&f) {
  if (lhs.size() != rhs.size()) {
    throw mk_runtime_error(fmt::format(
        "zip_with_strict requires inputs to have the same length, but received "
        "lhs = {} (length {}) and rhs = {} (length {})",
        lhs,
        lhs.size(),
        rhs,
        rhs.size()));
  }

  return zip_with(lhs, rhs, f);
}

} // namespace FlexFlow

#endif
