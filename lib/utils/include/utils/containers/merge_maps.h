#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_MAPS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_MAPS_H

#include "utils/containers/are_disjoint.h"
#include "utils/containers/keys.h"
#include "utils/containers/merge_method.dtg.h"
#include "utils/exception.h"
#include "utils/fmt/unordered_map.h"
#include "utils/fmt/unordered_set.h"
#include <unordered_map>

namespace FlexFlow {

template <typename K, typename V>
void merge_in_map(std::unordered_map<K, V> const &m,
                  std::unordered_map<K, V> &result) {
  for (auto const &[k, v] : m) {
    auto it = result.find(k);
    if (it != result.end()) {
      it->second = v;
    } else {
      result.insert({k, v});
    }
  }
}

template <typename K, typename V>
std::unordered_map<K, V>
    merge_disjoint_maps(std::unordered_map<K, V> const &lhs,
                        std::unordered_map<K, V> const &rhs) {

  std::unordered_set<K> lhs_keys = keys(lhs);
  std::unordered_set<K> rhs_keys = keys(rhs);
  std::unordered_set<K> shared_keys = intersection(lhs_keys, rhs_keys);
  if (!shared_keys.empty()) {
    throw mk_runtime_error(
        fmt::format("merge_maps expected disjoint maps, but maps share keys {}",
                    shared_keys));
  }

  std::unordered_map<K, V> result;
  merge_in_map(lhs, result);
  merge_in_map(rhs, result);
  return result;
}

template <typename K, typename V>
std::unordered_map<K, V>
    merge_map_left_dominates(std::unordered_map<K, V> const &lhs,
                             std::unordered_map<K, V> const &rhs) {
  std::unordered_map<K, V> result;
  merge_in_map(rhs, result);
  merge_in_map(lhs, result);
  return result;
}

template <typename K, typename V>
std::unordered_map<K, V>
    merge_map_right_dominates(std::unordered_map<K, V> const &lhs,
                              std::unordered_map<K, V> const &rhs) {
  std::unordered_map<K, V> result;
  merge_in_map(lhs, result);
  merge_in_map(rhs, result);
  return result;
}

} // namespace FlexFlow

#endif
