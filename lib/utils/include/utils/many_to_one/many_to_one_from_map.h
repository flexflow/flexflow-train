#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_FROM_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_FROM_MAP_H

#include "utils/many_to_one/many_to_one.h"

namespace FlexFlow {

template <typename L, typename R>
ManyToOne<L, R> many_to_one_from_map(std::unordered_map<L, R> const &m) {
  ManyToOne<L, R> result;

  for (auto const &[l, r] : m) {
    result.insert({l, r});
  }

  return result;
}

template <typename L, typename R>
ManyToOne<L, R> many_to_one_from_map(std::map<L, R> const &m) {
  ManyToOne<L, R> result;

  for (auto const &[l, r] : m) {
    result.insert({l, r});
  }

  return result;
}

} // namespace FlexFlow

#endif
