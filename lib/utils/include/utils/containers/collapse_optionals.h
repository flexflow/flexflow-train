#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_COLLAPSE_OPTIONALS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_COLLAPSE_OPTIONALS_H

#include <optional>

namespace FlexFlow {

template <typename T>
std::optional<T> collapse_optionals(std::optional<std::optional<T>> const &o) {
  if (!o.has_value()) {
    return std::nullopt;
  }

  return o.value();
}

} // namespace FlexFlow

#endif
