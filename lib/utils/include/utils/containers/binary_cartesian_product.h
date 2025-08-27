#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_CARTESIAN_PRODUCT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_BINARY_CARTESIAN_PRODUCT_H

#include <unordered_set>
#include "utils/hash/pair.h"

namespace FlexFlow {

template <typename A, typename B>
std::unordered_set<std::pair<A, B>>
  binary_cartesian_product(std::unordered_set<A> const &lhs,
                           std::unordered_set<B> const &rhs) {
  std::unordered_set<std::pair<A, B>> result;

  for (A const &a : lhs) {
    for (B const &b : rhs) {
      result.insert({a, b});
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
