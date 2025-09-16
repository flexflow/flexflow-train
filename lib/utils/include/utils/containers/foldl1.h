#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FOLDL1_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FOLDL1_H

#include <vector>
#include <libassert/assert.hpp>

namespace FlexFlow {

/**
 * @brief
 * Applies `func` to the elements of `c` from left to right, accumulating the
 * result. The first element of `c` is used as the starting point for the
 * accumulation.
 *
 * @example
 *   std::vector<int> nums = {1, 2, 3, 4};
 *   int result = foldl1(nums, [](int a, int b) { return a + b; });
 *   result -> (((1+2)+3)+4) = 10
 *
 * @note
 * For more information, see
 * https://hackage.haskell.org/package/base-4.20.0.1/docs/Prelude.html#v:foldl1
 * @throws std::runtime_error if the container is empty.
 */
template <typename C, typename F, typename E = typename C::value_type>
E foldl1(C const &c, F func) {
  ASSERT(!c.empty(),
        "foldl1 expected non-empty vector, but received empty vector");
  std::optional<E> result = std::nullopt;

  for (E const &e : c) {
    if (!result.has_value()) {
      result = e;
    } else {
      result = func(result.value(), e);
    }
  }
  return result.value();
}

} // namespace FlexFlow

#endif
