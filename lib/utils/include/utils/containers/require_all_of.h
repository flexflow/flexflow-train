#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_ALL_OF_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_REQUIRE_ALL_OF_H

#include <map>
#include <unordered_map>
#include <vector>

namespace FlexFlow {

template <typename C, typename F>
void require_all_of(C const &c, F &&f) {
  for (auto const &v : c) {
    f(v);
  }
}

template <typename K, typename V, typename F>
void require_all_of(std::unordered_map<K, V> const &m, F &&f) {
  for (auto const &[k, v] : m) {
    f(k, v);
  }
}

template <typename K, typename V, typename F>
void require_all_of(std::map<K, V> const &m, F &&f) {
  for (auto const &[k, v] : m) {
    f(k, v);
  }
}

} // namespace FlexFlow

#endif
