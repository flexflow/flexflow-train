#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_PAIRS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_TRANSFORM_PAIRS_H

#include "utils/containers/transform.h"

namespace FlexFlow {

template <typename F,
          typename L,
          typename R,
          typename Out = std::invoke_result_t<F, L, R>>
std::vector<Out> transform_pairs(std::vector<std::pair<L, R>> const &c, F &&f) {
  auto ff = [&](std::pair<L, R> const &p) -> Out {
    return f(p.first, p.second);
  };

  return transform(c, ff);
}

template <typename F,
          typename L,
          typename R,
          typename Out = std::invoke_result_t<F, L, R>>
std::unordered_set<Out> transform_pairs(std::unordered_set<std::pair<L, R>> const &c, F &&f) {
  auto ff = [&](std::pair<L, R> const &p) -> Out {
    return f(p.first, p.second);
  };

  return transform(c, ff);
}

template <typename F,
          typename L,
          typename R,
          typename Out = std::invoke_result_t<F, L, R>>
std::set<Out> transform_pairs(std::set<std::pair<L, R>> const &c, F &&f) {
  auto ff = [&](std::pair<L, R> const &p) -> Out {
    return f(p.first, p.second);
  };

  return transform(c, ff);
}

} // namespace FlexFlow

#endif
