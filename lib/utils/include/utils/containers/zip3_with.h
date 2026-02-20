#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP3_WITH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP3_WITH_H

#include <vector>

namespace FlexFlow {

template <typename F,
          typename A,
          typename B,
          typename C,
          typename Result = std::invoke_result_t<F, A, B, C>>
std::vector<Result> zip3_with(std::vector<A> const &v_a,
                              std::vector<B> const &v_b,
                              std::vector<C> const &v_c,
                              F &&f) {
  std::vector<Result> result;
  for (int i = 0; i < std::min(v_a.size(), std::min(v_b.size(), v_c.size()));
       i++) {
    result.push_back(f(v_a.at(i), v_b.at(i), v_c.at(i)));
  }

  return result;
}

} // namespace FlexFlow

#endif
