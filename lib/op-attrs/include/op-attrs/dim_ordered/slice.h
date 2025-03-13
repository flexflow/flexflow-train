#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_SLICE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_DIM_ORDERED_SLICE_H

#include "op-attrs/dim_ordered/dim_ordered.h"
#include "utils/containers/subvec.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/optional.h"

namespace FlexFlow {

template <typename Idx, typename T>
DimOrdered<Idx, T> nonoverloaded_slice(DimOrdered<Idx, T> const &d,
                                       std::optional<Idx> const &start,
                                       std::optional<Idx> const &end) {
  auto to_raw_idx = [](std::optional<Idx> const &idx) -> std::optional<int> {
    return transform(idx, [](Idx const &i) { return i.value; });
  };

  return DimOrdered<Idx, T>{
      subvec(vector_of(d), to_raw_idx(start), to_raw_idx(end))};
}
template <typename Idx, typename T>
DimOrdered<Idx, T> slice(DimOrdered<Idx, T> const &d,
                         std::optional<Idx> const &start = std::nullopt,
                         std::optional<Idx> const &end = std::nullopt) {
  return ff_dim_t_nonoverloaded_slice(d, start, end);
}

} // namespace FlexFlow

#endif
