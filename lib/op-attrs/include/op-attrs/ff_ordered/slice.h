#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_SLICE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_FF_ORDERED_SLICE_H

#include "op-attrs/ff_ordered/ff_ordered.h"
#include "utils/containers/slice.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

template <typename T>
FFOrdered<T> ff_dim_t_nonoverloaded_slice(FFOrdered<T> const &d,
                                          ff_dim_t const &start,
                                          std::optional<ff_dim_t> const &end) {
  int raw_start = start.value.unwrap_nonnegative();
  std::optional<int> raw_end = transform(
      end, [](ff_dim_t const &i) { return i.value.unwrap_nonnegative(); });
  return FFOrdered<T>{slice(vector_of(d), raw_start, raw_end)};
}

template <typename T>
FFOrdered<T> relative_ff_dim_t_nonoverloaded_slice(
    FFOrdered<T> const &d,
    relative_ff_dim_t const &start,
    std::optional<relative_ff_dim_t> const &end) {
  int raw_start = start.value;
  std::optional<int> raw_end =
      transform(end, [](relative_ff_dim_t const &i) { return i.value; });

  return FFOrdered<T>{slice(vector_of(d), raw_start, raw_end)};
}

template <typename T>
FFOrdered<T> slice(FFOrdered<T> const &d,
                   ff_dim_t const &start = ff_dim_t{0_n},
                   std::optional<ff_dim_t> const &end = std::nullopt) {
  return ff_dim_t_nonoverloaded_slice(d, start, end);
}

template <typename T>
FFOrdered<T> slice(FFOrdered<T> const &d,
                   relative_ff_dim_t const &start = relative_ff_dim_t{0},
                   std::optional<relative_ff_dim_t> const &end = std::nullopt) {
  return relative_ff_dim_t_nonoverloaded_slice(d, start, end);
}

} // namespace FlexFlow

#endif
