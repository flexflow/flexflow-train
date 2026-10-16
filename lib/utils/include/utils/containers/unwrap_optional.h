#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNWRAP_OPTIONAL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNWRAP_OPTIONAL_H

#include "utils/type_traits_core.h"
#include <optional>

namespace FlexFlow {

template <typename T>
struct unwrap_optional {
  static_assert("T is not a std::optional!");
};

template <typename T>
struct unwrap_optional<std::optional<T>> : type_identity<T> {};

template <typename T>
using unwrap_optional_t = typename unwrap_optional<T>::type;

} // namespace FlexFlow

#endif
