#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_ORTHOTOPE_DIM_INDEXED_FROM_IDX_MAP_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_ORTHOTOPE_DIM_INDEXED_ORTHOTOPE_DIM_INDEXED_FROM_IDX_MAP_H

#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed.h"
#include "utils/orthotope/orthotope_dim_indexed/orthotope_dim_indexed_of.h"
#include "utils/containers/vector_from_idx_map.h"
#include "utils/containers/map_keys.h"

namespace FlexFlow {

template <typename T>
std::optional<OrthotopeDimIndexed<T>> orthotope_dim_indexed_from_idx_map(std::unordered_map<orthotope_dim_idx_t, T> const &m) {
  std::unordered_map<int, T> raw_idx_map = map_keys(m, [](orthotope_dim_idx_t idx) { return idx.raw_idx; });

  std::vector<T> raw_vec = ({
    std::optional<std::vector<T>> returned = vector_from_idx_map(raw_idx_map);
    if (!returned.has_value()) {
      return std::nullopt;
    }

    returned.value();
  });

  return orthotope_dim_indexed_of(raw_vec);
}

} // namespace FlexFlow

#endif
