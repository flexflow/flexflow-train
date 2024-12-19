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

template <typename T>
FFOrdered<T> ff_dim_t_nonoverloaded_slice(FFOrdered<T> const &d,
                                          std::optional<ff_dim_t> const &start,
                                          std::optional<ff_dim_t> const &end) {
  auto to_raw_idx =
      [](std::optional<ff_dim_t> const &idx) -> std::optional<int> {
    return transform(idx,
                     [](ff_dim_t const &i) { return i.value.get_value(); });
  };

  return FFOrdered<T>{subvec(vector_of(d), to_raw_idx(start), to_raw_idx(end))};
}

template <typename T>
FFOrdered<T> relative_ff_dim_t_nonoverloaded_slice(
    FFOrdered<T> const &d,
    std::optional<relative_ff_dim_t> const &start,
    std::optional<relative_ff_dim_t> const &end) {
  auto to_raw_idx =
      [](std::optional<relative_ff_dim_t> const &idx) -> std::optional<int> {
    return transform(idx, [](relative_ff_dim_t const &i) { return i.value; });
  };

  return FFOrdered<T>{subvec(vector_of(d), to_raw_idx(start), to_raw_idx(end))};
}

template <typename Idx, typename T>
DimOrdered<Idx, T> slice(DimOrdered<Idx, T> const &d,
                         std::nullopt_t const &start,
                         Idx const &end) {
  return nonoverloaded_slice(
      d, std::optional<Idx>{start}, std::optional<Idx>{end});
}

template <typename Idx, typename T>
DimOrdered<Idx, T> slice(DimOrdered<Idx, T> const &d,
                         Idx const &start,
                         std::nullopt_t const &end) {
  return nonoverloaded_slice(
      d, std::optional<Idx>{start}, std::optional<Idx>{end});
}

template <typename Idx, typename T>
DimOrdered<Idx, T>
    slice(DimOrdered<Idx, T> const &d, Idx const &start, Idx const &end) {
  return nonoverloaded_slice(
      d, std::optional<Idx>{start}, std::optional<Idx>{end});
}

template <typename T>
FFOrdered<T> slice(FFOrdered<T> const &d,
                   std::nullopt_t const &start,
                   ff_dim_t const &end) {
  return ff_dim_t_nonoverloaded_slice(
      d, std::optional<ff_dim_t>{start}, std::optional<ff_dim_t>{end});
}

template <typename T>
FFOrdered<T> slice(FFOrdered<T> const &d,
                   ff_dim_t const &start,
                   std::nullopt_t const &end) {
  return ff_dim_t_nonoverloaded_slice(
      d, std::optional<ff_dim_t>{start}, std::optional<ff_dim_t>{end});
}

template <typename T>
FFOrdered<T>
    slice(FFOrdered<T> const &d, ff_dim_t const &start, ff_dim_t const &end) {
  return ff_dim_t_nonoverloaded_slice(
      d, std::optional<ff_dim_t>{start}, std::optional<ff_dim_t>{end});
}

template <typename T>
FFOrdered<T> slice(FFOrdered<T> const &d,
                   std::nullopt_t const &start,
                   relative_ff_dim_t const &end) {
  return relative_ff_dim_t_nonoverloaded_slice(
      d,
      std::optional<relative_ff_dim_t>{start},
      std::optional<relative_ff_dim_t>{end});
}

template <typename T>
FFOrdered<T> slice(FFOrdered<T> const &d,
                   relative_ff_dim_t const &start,
                   std::nullopt_t const &end) {
  return relative_ff_dim_t_nonoverloaded_slice(
      d,
      std::optional<relative_ff_dim_t>{start},
      std::optional<relative_ff_dim_t>{end});
}

template <typename T>
FFOrdered<T> slice(FFOrdered<T> const &d,
                   relative_ff_dim_t const &start,
                   relative_ff_dim_t const &end) {
  return relative_ff_dim_t_nonoverloaded_slice(
      d,
      std::optional<relative_ff_dim_t>{start},
      std::optional<relative_ff_dim_t>{end});
}

} // namespace FlexFlow

#endif
