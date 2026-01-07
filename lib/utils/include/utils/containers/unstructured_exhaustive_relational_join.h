#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNSTRUCTURED_EXHAUSTIVE_RELATIONAL_JOIN_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_UNSTRUCTURED_EXHAUSTIVE_RELATIONAL_JOIN_H

#include "utils/containers/transform.h"
#include "utils/hash/pair.h"
#include <libassert/assert.hpp>
#include <unordered_set>

namespace FlexFlow {

template <typename L, typename C, typename R>
std::unordered_set<std::pair<L, R>> unstructured_exhaustive_relational_join(
    std::unordered_set<std::pair<L, C>> const &lhs,
    std::unordered_set<std::pair<C, R>> const &rhs) {
  std::unordered_set<std::pair<L, R>> result;

  std::unordered_set<L> lhs_ls =
      transform(lhs, [](std::pair<L, C> const &lc) { return lc.first; });
  std::unordered_set<C> lhs_cs =
      transform(lhs, [](std::pair<L, C> const &lc) { return lc.second; });
  std::unordered_set<C> rhs_cs =
      transform(rhs, [](std::pair<C, R> const &cr) { return cr.first; });
  std::unordered_set<R> rhs_rs =
      transform(rhs, [](std::pair<C, R> const &cr) { return cr.second; });

  ASSERT(lhs_cs == rhs_cs);

  std::unordered_set<L> result_ls;
  std::unordered_set<R> result_rs;

  for (auto const &[l, c1] : lhs) {
    for (auto const &[c2, r] : rhs) {
      if (c1 == c2) {
        result.insert({l, r});
        result_ls.insert(l);
        result_rs.insert(r);
      }
    }
  }

  ASSERT(result_ls == lhs_ls);
  ASSERT(result_rs == rhs_rs);

  return result;
}

} // namespace FlexFlow

#endif
