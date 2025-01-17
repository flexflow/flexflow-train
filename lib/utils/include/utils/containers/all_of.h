#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ALL_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_ALL_OF_H

#include <vector>
#include <unordered_map>
#include <map>

namespace FlexFlow {

template <typename C, typename F>
bool all_of(C const &c, F &&f) {
  for (auto const &v : c) {
    if (!f(v)) {
      return false;
    }
  }
  return true;
}

template <typename K, typename V, typename F>
bool all_of(std::unordered_map<K, V> const &m, F &&f) {
  for (auto const &[k, v] : m) {
    if (!f(k, v)) {
      return false;
    }
  }

  return true;
}

template <typename K, typename V, typename F>
bool all_of(std::map<K, V> const &m, F &&f) {
  for (auto const &[k, v] : m) {
    if (!f(k, v)) {
      return false;
    }
  }

  return true;
}

bool all_of(std::vector<bool> const &);

} // namespace FlexFlow

#endif
