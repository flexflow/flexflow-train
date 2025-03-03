#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP3_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP3_H

#include <algorithm>
#include <set>
#include <tuple>
#include <vector>

namespace FlexFlow {

template <typename A, typename B, typename C>
std::vector<std::tuple<A, B, C>> zip3(std::vector<A> const &a,
                                      std::vector<B> const &b,
                                      std::vector<C> const &c) {
  std::vector<std::tuple<A, B, C>> result;
  for (int i = 0; i < std::min({a.size(), b.size(), c.size()}); i++) {
    result.push_back(std::make_tuple(a.at(i), b.at(i), c.at(i)));
  }
  return result;
}

} // namespace FlexFlow

#endif
