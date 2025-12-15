#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_FROM_UNSTRUCTURED_RELATION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_FROM_UNSTRUCTURED_RELATION_H

#include "utils/many_to_one/many_to_one.h"

namespace FlexFlow {

template <typename L, typename R>
ManyToOne<L, R> many_to_one_from_unstructured_relation(
  std::unordered_set<std::pair<L, R>> const &relation)
{
  ManyToOne<L, R> result; 
  for (auto const &lr : relation) {
    result.insert(lr);
  }
  return result;
}

} // namespace FlexFlow

#endif
