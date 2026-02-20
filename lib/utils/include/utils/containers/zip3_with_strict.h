#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP3_WITH_STRICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP3_WITH_STRICT_H

#include "utils/containers/zip3_with.h"
#include <libassert/assert.hpp>
#include <vector>

namespace FlexFlow {

template <typename F,
          typename A,
          typename B,
          typename C,
          typename Result = std::invoke_result_t<F, A, B, C>>
std::vector<Result> zip3_with_strict(std::vector<A> const &v_a,
                                     std::vector<B> const &v_b,
                                     std::vector<C> const &v_c,
                                     F &&f) {
  ASSERT(v_a.size() == v_b.size() && v_b.size() == v_c.size(),
         "zip3_with_strict requires inputs to have the same length, but "
         "received mismatched lengths",
         v_a.size(),
         v_b.size(),
         v_c.size());

  return zip3_with(v_a, v_b, v_c, f);
}

} // namespace FlexFlow

#endif
