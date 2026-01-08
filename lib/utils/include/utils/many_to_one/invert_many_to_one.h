#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_INVERT_MANY_TO_ONE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_INVERT_MANY_TO_ONE_H

#include "utils/many_to_one/many_to_one.h"
#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename L, typename R>
OneToMany<R, L> invert_many_to_one(ManyToOne<L, R> const &many_to_one) {
  OneToMany<R, L> result;

  for (L const &l : many_to_one.left_values()) {
    result.insert({many_to_one.at_l(l), l});
  }

  return result;
}

} // namespace FlexFlow

#endif
