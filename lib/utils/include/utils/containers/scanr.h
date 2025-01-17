#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SCANR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SCANR_H

#include <vector>
#include "utils/containers/reversed.h"

namespace FlexFlow {

/**
 * @brief
 * Applies `f` to the elements of `c` from right to left, accumulating
 * the intermediate results in a vector. `init` is used as the starting point
 * for the accumulation.
 *
 * @example
 *   std::vector<int> nums = {1, 2, 3, 4};
 *   auto result = scanl(nums, 0, [](int a, int b) {return a+b;});
 *   result -> {10, 9, 7, 4, 0}
 *
 * @note
 * Essentially a foldl which stores the intermediate results
 * For more information, see
 * https://hackage.haskell.org/package/base-4.20.0.1/docs/Prelude.html#v:scan4
 */
template <typename C, typename F, typename T>
std::vector<T> scanr(C const &c, T init, F &&f) {
  std::vector<T> result;

  result.push_back(init);
  for (auto it = c.crbegin(); it != c.crend(); it++) {
    init = f(*it, init);
    result.push_back(init);
  }

  return reversed(result);
}

} // namespace FlexFlow

#endif
