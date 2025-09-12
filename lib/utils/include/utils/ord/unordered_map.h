#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORD_UNORDERED_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORD_UNORDERED_MAP_H

#include "utils/type_traits_core.h"
#include <algorithm>
#include <unordered_map>
#include <map>

namespace FlexFlow {

template <typename K, typename V>
std::enable_if_t<is_lt_comparable_v<std::pair<K, V>>, bool>
    operator<(std::unordered_map<K, V> const &lhs, std::unordered_map<K, V> const &rhs) {
  CHECK_LT_COMPARABLE(K);
  CHECK_LT_COMPARABLE(V);

  std::map<K, V> lhs_ordered(lhs.cbegin(), lhs.cend());
  std::map<K, V> rhs_ordered(rhs.cbegin(), rhs.cend());

  return lhs_ordered < rhs_ordered;
}


} // namespace FlexFlow

#endif
