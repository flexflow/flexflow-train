#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_REQUIRE_ONE_TO_MANY_IS_BIJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_REQUIRE_ONE_TO_MANY_IS_BIJECTION_H

#include "utils/bidict/algorithms/bidict_from_map.h"
#include "utils/containers/map_values.h"
#include "utils/containers/get_only.h"
#include "utils/one_to_many/one_to_many.h"
#include "utils/nonempty_set/nonempty_set.h"

namespace FlexFlow {

template <typename L, typename R>
bidict<L, R> require_one_to_many_is_bijection(OneToMany<L, R> const &otm) {
  return bidict_from_map(
    map_values(otm.l_to_r(),
               [](nonempty_set<R> const &s) -> R {
                 return get_only(s.unwrap_as_set());
               }));
}

} // namespace FlexFlow

#endif
