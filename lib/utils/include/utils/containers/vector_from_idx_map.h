#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VECTOR_FROM_IDX_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_CONTAINERS_VECTOR_FROM_IDX_MAP_H

#include "utils/containers/contains_key.h"
#include "utils/nonnegative_int/nonnegative_int.h"
#include <optional>
#include <unordered_map>
#include <vector>

namespace FlexFlow {

template <typename T>
std::optional<std::vector<T>>
    vector_from_idx_map(std::unordered_map<nonnegative_int, T> const &m) {
  std::vector<T> result;

  for (nonnegative_int i = 0_n; i < m.size(); i++) {
    if (!contains_key(m, i)) {
      return std::nullopt;
    }
    result.push_back(m.at(i));
  }

  return result;
}

} // namespace FlexFlow

#endif
