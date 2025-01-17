#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_FROM_L_TO_R_MAPPING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_FROM_L_TO_R_MAPPING_H

#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename L, typename R>
OneToMany<L, R> one_to_many_from_l_to_r_mapping(std::unordered_map<L, std::unordered_set<R>> const &m) {
  OneToMany<L, R> result;

  for (auto const &[l, rs] : m) {
    for (auto const &r : rs) {
      result.insert({l, r});
    }
  }

  return result;
}

} // namespace FlexFlow

#endif
