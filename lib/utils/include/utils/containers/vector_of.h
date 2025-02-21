#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VECTOR_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VECTOR_OF_H

#include <optional>
#include <vector>

namespace FlexFlow {

template <typename C, typename E = typename C::value_type>
std::vector<E> vector_of(C const &c) {
  std::vector<E> result(c.cbegin(), c.cend());
  return result;
}

template <typename T>
std::vector<T> vector_of(std::optional<T> const &o) {
  if (o.has_value()) {
    return {o.value()};
  } else {
    return {};
  }
}

} // namespace FlexFlow

#endif
