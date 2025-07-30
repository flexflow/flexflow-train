#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CONTAINS_DUPLICATES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_CONTAINS_DUPLICATES_H

#include "utils/containers/unordered_set_of.h"
#include <set>
#include <vector>

namespace FlexFlow {

template <typename T>
bool contains_duplicates(std::vector<T> const &s) {
  return unordered_set_of(s).size() != s.size();
}

template <typename T>
bool contains_duplicates(std::unordered_multiset<T> const &s) {
  return unordered_set_of(s).size() != s.size();
}

template <typename T>
bool contains_duplicates(std::multiset<T> const &s) {
  return unordered_set_of(s).size() != s.size();
}

} // namespace FlexFlow

#endif
