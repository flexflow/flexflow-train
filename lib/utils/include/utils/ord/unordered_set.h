#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORD_UNORDERED_SET_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORD_UNORDERED_SET_H

#include "utils/type_traits_core.h"
#include <set>
#include <unordered_set>

namespace FlexFlow {

template <typename T>
std::enable_if_t<is_lt_comparable_v<T>, bool>
    operator<(std::unordered_set<T> const &lhs,
              std::unordered_set<T> const &rhs) {
  CHECK_LT_COMPARABLE(T);

  std::set<T> lhs_ordered(lhs.cbegin(), lhs.cend());
  std::set<T> rhs_ordered(rhs.cbegin(), rhs.cend());

  return lhs_ordered < rhs_ordered;
}

} // namespace FlexFlow

#endif
