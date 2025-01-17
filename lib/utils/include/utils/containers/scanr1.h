#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SCANR1_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SCANR1_H

#include <vector>
#include <optional>
#include "utils/containers/reversed.h"

namespace FlexFlow {

/**
 * @brief
 * Applies `op` to the elements of `c` from right to left, accumulating
 * the intermediate results in a vector. The first item of `c` is used as the
 * starting point for the accumulation.
 *
 * @example
 *   std::vector<int> nums = {1, 2, 3, 4};
 *   auto result = scanl1(nums, [](int a, int b) {return a+b;});
 *   result -> {10, 9, 7, 4}
 *
 * @note
 * Essentially a foldr1 which stores the intermediate results.
 * For more information, see
 * https://hackage.haskell.org/package/base-4.20.0.1/docs/Prelude.html#v:scanl1
 */
template <typename C, typename F, typename T = typename C::value_type>
std::vector<T> scanr1(C const &c, F &&f) {

  if (c.empty()) {
    return std::vector<T>();
  }

  std::optional<T> init = std::nullopt;
  std::vector<T> result;

  for (auto it = c.crbegin(); it != c.crend(); it++) {
    if (!init.has_value()) {
      init = *it;
    } else {
      init = f(*it, init.value());
    }
    result.push_back(init.value());
  }
  return reversed(result);
}

} // namespace FlexFlow

#endif
