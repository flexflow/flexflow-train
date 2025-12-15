#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_UNSTRUCTURED_RELATION_FROM_MANY_TO_ONE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_UNSTRUCTURED_RELATION_FROM_MANY_TO_ONE_H

#include "utils/many_to_one/many_to_one.h"
#include "utils/containers/unordered_set_of.h"

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<std::pair<L, R>> 
  unstructured_relation_from_many_to_one(
    ManyToOne<L, R> const &many_to_one) 
{
  return unordered_set_of(many_to_one.l_to_r());
}

} // namespace FlexFlow

#endif
