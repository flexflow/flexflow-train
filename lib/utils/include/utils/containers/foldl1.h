#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FOLDL1_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_FOLDL1_H

#include "utils/containers/get_element_type.h"
#include "utils/containers/is_empty.h"
#include <libassert/assert.hpp>
#include <vector>

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
template <typename C, typename F, typename E = get_element_type_t<C>>
E foldl1(C const &c, F func) {
  ASSERT(!is_empty(c),
         "foldl1 expected non-empty container, but received empty container");
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
