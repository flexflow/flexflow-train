#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_UNSTRUCTURED_RELATION_FROM_ONE_TO_MANY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_UNSTRUCTURED_RELATION_FROM_ONE_TO_MANY_H

#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<std::pair<L, R>>
    unstructured_relation_from_one_to_many(OneToMany<L, R> const &one_to_many) {
  return transform(unordered_set_of(one_to_many.r_to_l()),
                   [](std::pair<R, L> const &rl) -> std::pair<L, R> {
                     return std::pair{rl.second, rl.first};
                   });
}

} // namespace FlexFlow

#endif
