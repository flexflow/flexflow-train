#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_FROM_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_FROM_BIDICT_H

#include "utils/bidict/bidict.h"
#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename L, typename R>
OneToMany<L, R> one_to_many_from_bidict(bidict<L, R> const &b) {
  OneToMany<L, R> result;

  for (auto const &[l, r] : b) {
    result.insert({l, r});
  }

  return result;
}

} // namespace FlexFlow

#endif
