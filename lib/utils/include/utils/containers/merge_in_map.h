#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_IN_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_MERGE_IN_MAP_H

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

} // namespace FlexFlow

#endif
