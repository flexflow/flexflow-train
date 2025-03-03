#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_STRICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_STRICT_H

#include "utils/containers/zip.h"
#include "utils/exception.h"
#include "utils/fmt/vector.h"

namespace FlexFlow {

template <typename L, typename R>
std::vector<std::pair<L, R>> zip_strict(std::vector<L> const &lhs,
                                        std::vector<R> const &rhs) {
  if (lhs.size() != rhs.size()) {
    throw mk_runtime_error(
        fmt::format("zip_strict requires lhs and rhs to have the same length, "
                    "but received lhs={} (length {}), rhs={} (length {})",
                    lhs,
                    lhs.size(),
                    rhs,
                    rhs.size()));
  }

  return zip(lhs, rhs);
}

} // namespace FlexFlow

#endif
