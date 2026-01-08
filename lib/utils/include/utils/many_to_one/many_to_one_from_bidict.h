#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_FROM_BIDICT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_MANY_TO_ONE_MANY_TO_ONE_FROM_BIDICT_H

#include "utils/bidict/bidict.h"
#include "utils/many_to_one/many_to_one.h"

namespace FlexFlow {

template <typename L, typename R>
ManyToOne<L, R> many_to_one_from_bidict(bidict<L, R> const &b) {
  ManyToOne<L, R> result;

  for (auto const &[l, r] : b) {
    result.insert({l, r});
  }

  return result;
}

} // namespace FlexFlow

#endif
