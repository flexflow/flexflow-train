#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_UNTIL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_UNTIL_H

#include <optional>
#include <type_traits>
#include <vector>

namespace FlexFlow {

template <typename T,
          typename F,
          typename T2 = typename std::invoke_result_t<F, T>::value_type>
std::vector<T2> transform_until(std::vector<T> const &ts, F &&f) {
  std::vector<T2> result;

  for (T const &t : ts) {
    std::optional<T2> x = f(t);

    if (x.has_value()) {
      result.push_back(x.value());
    } else {
      break;
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
