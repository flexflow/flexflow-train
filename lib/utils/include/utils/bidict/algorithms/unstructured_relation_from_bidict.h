#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_UNSTRUCTURED_RELATION_FROM_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_BIDICT_ALGORITHMS_UNSTRUCTURED_RELATION_FROM_BIDICT_H

#include "utils/bidict/bidict.h"
#include "utils/bidict/algorithms/unordered_set_of.h"

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<std::pair<L, R>>
  unstructured_relation_from_bidict(bidict<L, R> const &b)
{
  return unordered_set_of(b);
}

} // namespace FlexFlow

#endif
