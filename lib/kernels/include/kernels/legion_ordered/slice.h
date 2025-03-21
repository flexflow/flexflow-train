#ifndef _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LEGION_ORDERED_SLICE_H
#define _FLEXFLOW_LIB_KERNELS_INCLUDE_KERNELS_LEGION_ORDERED_SLICE_H

#include "kernels/legion_ordered/legion_ordered.h"
#include "utils/containers/slice.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

template <typename T>
LegionOrdered<T> slice(LegionOrdered<T> const &d,
                       legion_dim_t const &start,
                       std::optional<legion_dim_t> const &end) {
  int raw_start = start.value.unwrap_nonnegative();
  std::optional<int> raw_end = transform(
      end, [](legion_dim_t const &i) { return i.value.unwrap_nonnegative(); });

  return LegionOrdered<T>{slice(vector_of(d), raw_start, raw_end)};
}

} // namespace FlexFlow

#endif
