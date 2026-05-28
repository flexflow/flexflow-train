#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_FILTER_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_FILTER_VALUES_H

#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename L, typename R, typename F>
OneToMany<L, R> one_to_many_filter_values(OneToMany<L, R> const &m, F &&f) {
  OneToMany<L, R> result;
  for (auto const &kv : unstructured_relation_from_one_to_many(m)) {
    if (f(kv.second)) {
      result.insert(kv);
    }
  }
  return result;
}

} // namespace FlexFlow


#endif
