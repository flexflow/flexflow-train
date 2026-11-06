#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ELEMENT_COUNTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GET_ELEMENT_COUNTS_H

#include "utils/containers/contains_key.h"
#include "utils/positive_int/positive_int.h"
#include <map>
#include <set>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename T>
std::map<T, positive_int> get_element_counts(std::vector<T> const &v) {
  std::map<T, positive_int> counts;
  for (T const &t : v) {
    if (!contains_key(counts, t)) {
      counts.insert({t, 1_p});
    } else {
      counts.at(t)++;
    }
  }
  return counts;
}

template <typename T>
std::map<T, positive_int> get_element_counts(std::multiset<T> const &v) {
  std::map<T, positive_int> counts;
  for (T const &t : v) {
    if (!contains_key(counts, t)) {
      counts.insert({t, 1_p});
    } else {
      counts.at(t)++;
    }
  }
  return counts;
}

template <typename T>
std::unordered_map<T, positive_int> get_element_counts(std::unordered_multiset<T> const &v) {
  std::unordered_map<T, positive_int> counts;
  for (T const &t : v) {
    if (!contains_key(counts, t)) {
      counts.insert({t, 1_p});
    } else {
      counts.at(t)++;
    }
  }
  return counts;
}

std::map<char, positive_int> get_element_counts(std::string const &);

} // namespace FlexFlow

#endif
