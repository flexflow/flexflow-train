#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_WITH_STRICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_WITH_STRICT_H

#include "utils/containers/zip_with.h"
#include "utils/fmt/vector.h"
#include <vector>
#include <libassert/assert.hpp>

namespace FlexFlow {

template <typename T1,
          typename T2,
          typename F,
          typename Result = std::invoke_result_t<F, T1, T2>>
std::vector<Result> zip_with_strict(std::vector<T1> const &lhs,
                                    std::vector<T2> const &rhs,
                                    F &&f) {
  ASSERT(lhs.size() == rhs.size(),
         "zip_with_strict requires inputs to have the same length."
         "For a similar function without this requirement, see zip_with.");

  return zip_with(lhs, rhs, f);
}

} // namespace FlexFlow

#endif
