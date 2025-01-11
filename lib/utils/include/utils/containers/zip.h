#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ZIP_H

#include <tuple>
#include <utility>
#include <vector>
#include "utils/exception.h"
#include "fmt/format.h"

namespace FlexFlow {

template <typename L, typename R>
std::vector<std::pair<L, R>> zip(std::vector<L> const &l,
                                std::vector<R> const &r,
                                bool strict = false) {
  if (strict && l.size() != r.size()) {
    throw mk_runtime_error(fmt::format(
        "When strict = true, vector sizes must match. Got vectors of length {} and {}",
        l.size(), r.size()));
  }

  std::vector<std::pair<L, R>> result;
  for (int i = 0; i < std::min(l.size(), r.size()); i++) {
    result.push_back(std::make_pair(l.at(i), r.at(i)));
  }
  return result;
}

template <typename A, typename B, typename C>
std::vector<std::tuple<A, B, C>> zip(std::vector<A> const &a,
                                    std::vector<B> const &b,
                                    std::vector<C> const &c,
                                    bool strict = false) {
  if (strict && (a.size() != b.size() || b.size() != c.size())) {
    throw std::runtime_error(fmt::format(
        "When strict = true, vectors sizes must match. Got vectors of length {}, {} and {}",
        a.size(), b.size(), c.size()));
  }
  
  std::vector<std::tuple<A, B, C>> result;
  for (int i = 0; i < std::min({a.size(), b.size(), c.size()}); i++) {
    result.push_back(std::make_tuple(a.at(i), b.at(i), c.at(i)));
  }
  return result;
}

} // namespace FlexFlow

#endif
