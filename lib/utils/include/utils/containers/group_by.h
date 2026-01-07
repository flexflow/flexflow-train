#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GROUP_BY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_GROUP_BY_H

#include "utils/one_to_many/one_to_many.h"
#include <set>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace FlexFlow {

template <typename V, typename F, typename K = std::invoke_result_t<F, V>>
OneToMany<K, V> group_by(std::unordered_set<V> const &vs, F &&f) {
  OneToMany<K, V> result;
  for (V const &v : vs) {
    result.insert({f(v), v});
  }
  return result;
}

template <typename V, typename F, typename K = std::invoke_result_t<F, V>>
OneToMany<K, V> group_by(std::set<V> const &vs, F &&f) {
  OneToMany<K, V> result;
  for (V const &v : vs) {
    result.insert({f(v), v});
  }
  return result;
}

template <typename V, typename F, typename K = std::invoke_result_t<F, V>>
std::unordered_map<K, std::vector<V>> group_by(std::vector<V> const &vs,
                                               F &&f) {
  std::unordered_map<K, std::vector<V>> result;
  for (V const &v : vs) {
    result[f(v)].push_back(v);
  }
  return result;
}

} // namespace FlexFlow

#endif
