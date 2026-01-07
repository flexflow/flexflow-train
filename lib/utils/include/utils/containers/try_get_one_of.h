#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRY_GET_ONE_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRY_GET_ONE_OF_H

#include <optional>
#include <set>
#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::optional<T> try_get_one_of(std::unordered_set<T> const &s) {
  if (s.empty()) {
    return std::nullopt;
  } else {
    return *s.cbegin();
  }
}

template <typename T>
std::optional<T> try_get_one_of(std::set<T> const &s) {
  if (s.empty()) {
    return std::nullopt;
  } else {
    return *s.cbegin();
  }
}

} // namespace FlexFlow

#endif
