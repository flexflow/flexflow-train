#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_IS_SUBSETEQ_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_IS_SUBSETEQ_OF_H

#include "utils/containers/contains.h"
#include <unordered_set>
#include <set>

namespace FlexFlow {

template <typename T>
bool is_subseteq_of(std::unordered_set<T> const &sub,
                    std::unordered_set<T> const &super) {
  if (sub.size() > super.size()) {
    return false;
  }

  for (auto const &s : sub) {
    if (!contains(super, s)) {
      return false;
    }
  }
  return true;
}

template <typename T>
bool is_subseteq_of(std::set<T> const &l,
                    std::set<T> const &r) {
  if (l.size() > r.size()) {
    return false;
  }

  for (auto const &ll : l) {
    if (!contains(r, ll)) {
      return false;
    }
  }
  return true;
}

} // namespace FlexFlow

#endif
