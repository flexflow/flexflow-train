#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORD_VECTOR_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORD_VECTOR_H

#include "utils/type_traits_core.h"
#include <algorithm>
#include <vector>

namespace FlexFlow {

template <typename T>
std::enable_if_t<is_lt_comparable_v<T>, bool> operator<(std::vector<T> const &lhs, std::vector<T> const &rhs) {
  return std::lexicographical_compare(lhs.cbegin(), lhs.cend(), rhs.cbegin(), rhs.cend());
}

} // namespace FlexFlow

#endif
