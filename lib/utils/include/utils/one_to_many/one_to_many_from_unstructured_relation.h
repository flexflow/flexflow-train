#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_FROM_UNSTRUCTURED_RELATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_FROM_UNSTRUCTURED_RELATION_H

#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename L, typename R>
OneToMany<L, R> one_to_many_from_unstructured_relation(
    std::unordered_set<std::pair<L, R>> const &rel) {
  OneToMany<L, R> result;
  for (auto const &lr : rel) {
    result.insert(lr);
  }
  return result;
}

} // namespace FlexFlow

#endif
