#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_INVERT_ONE_TO_MANY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_INVERT_ONE_TO_MANY_H

#include "utils/many_to_one/many_to_one.h"
#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename L, typename R>
ManyToOne<R, L> invert_one_to_many(OneToMany<L, R> const &one_to_many) {
  ManyToOne<R, L> result;

  for (R const &r : one_to_many.right_values()) {
    result.insert({r, one_to_many.at_r(r)});
  }

  return result;
}

} // namespace FlexFlow

#endif
