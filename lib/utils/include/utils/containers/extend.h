#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_EXTEND_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_EXTEND_H

#include "utils/containers/extend_vector.h"
#include <set>
#include <unordered_set>

namespace FlexFlow {

template <typename T, typename C>
void extend(std::vector<T> &lhs, C const &rhs) {
  extend_vector(lhs, rhs);
}

template <typename T, typename C>
void extend(std::unordered_set<T> &lhs, C const &rhs) {
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(rhs.cbegin(), rhs.cend());
}

template <typename T, typename C>
void extend(std::unordered_multiset<T> &lhs, C const &rhs) {
  lhs.reserve(lhs.size() + std::distance(rhs.begin(), rhs.end()));
  lhs.insert(rhs.cbegin(), rhs.cend());
}

template <typename T, typename C>
void extend(std::set<T> &lhs, C const &rhs) {
  lhs.insert(rhs.cbegin(), rhs.cend());
}

template <typename T, typename C>
void extend(std::multiset<T> &lhs, C const &rhs) {
  lhs.insert(rhs.cbegin(), rhs.cend());
}

} // namespace FlexFlow

#endif
