#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VECTOR_FROM_IDX_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VECTOR_FROM_IDX_MAP_H

#include <optional>
#include <unordered_map>
#include "utils/containers/contains_key.h"
#include <vector>

namespace FlexFlow {

template <typename T>
std::optional<std::vector<T>> vector_from_idx_map(std::unordered_map<int, T> const &m) {
  std::vector<T> result;

  for (int i = 0; i < m.size(); i++) {
    if (!contains_key(m, i)) {
      return std::nullopt;
    }
    result.push_back(m.at(i));
  }

  return result;
}

} // namespace FlexFlow

#endif
