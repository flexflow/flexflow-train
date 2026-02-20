#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SCANL_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_SCANL_H

#include <vector>

namespace FlexFlow {

/**
 * @brief
 * Applies `f` to the elements of `c` from left to right, accumulating
 * the intermediate results in a vector. `init` is used as the starting point
 * for the accumulation.
 *
 * @example
 *   std::vector<int> nums = {1, 2, 3, 4};
 *   auto result = scanl(nums, 0, [](int a, int b) {return a+b;});
 *   result -> {0,1,3,6,10}
 *
 * @note
 * Essentially a foldl which stores the intermediate results
 * For more information, see
 * https://hackage.haskell.org/package/base-4.20.0.1/docs/Prelude.html#v:scanl
 */
template <typename C, typename F, typename T>
std::vector<T> scanl(C const &c, T init, F &&f) {
  std::vector<T> result;

  result.push_back(init);
  for (auto const &elem : c) {
    init = f(init, elem);
    result.push_back(init);
  }

  return result;
}

} // namespace FlexFlow

#endif
