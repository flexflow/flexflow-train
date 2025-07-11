#include "op-attrs/tensor_dims.h"
#include "op-attrs/ff_ordered/get_idxs.h"
#include "op-attrs/ff_ordered/slice.h"
#include "op-attrs/ff_ordered/zip.h"
#include "op-attrs/ff_ordered/zip_with.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.dtg.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/all_of.h"
#include "utils/containers/cartesian_product.h"
#include "utils/containers/contains.h"
#include "utils/containers/product.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/zip.h"
#include "utils/integer_conversions.h"
#include "utils/nonnegative_int/nonnegative_range.h"
#include "utils/nonnegative_int/num_elements.h"
#include "op-attrs/ff_ordered/enumerate.h"
#include "op-attrs/ff_ordered/filtrans.h"

namespace FlexFlow {

FFOrdered<positive_int> const &ff_ordered(TensorDims const &dims) {
  return dims.ff_ordered;
}

bool tensor_dims_has_dim(TensorDims const &tensor_dims, ff_dim_t dim) {
  return contains(get_idxs(tensor_dims.ff_ordered), dim);
}

nonnegative_int get_num_dims(TensorDims const &dims) {
  return num_elements(dims.ff_ordered);
}

positive_int dim_at_idx(TensorDims const &dims, relative_ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

positive_int &dim_at_idx(TensorDims &dims, relative_ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

positive_int dim_at_idx(TensorDims const &dims, ff_dim_t ff_dim_idx) {
  return dims.ff_ordered.at(ff_dim_idx);
}

positive_int &dim_at_idx(TensorDims &dims, ff_dim_t ff_dim_idx) {
  return dims.ff_ordered.at(ff_dim_idx);
}

std::optional<positive_int> try_dim_at_idx(TensorDims const &dims, relative_ff_dim_t idx) {
  if (dims.ff_ordered.idx_is_valid(idx)) {
    return dims.ff_ordered.at(idx);
  } else {
    return std::nullopt;
  }
}

std::optional<positive_int> try_dim_at_idx(TensorDims const &dims, ff_dim_t idx) {
  if (dims.ff_ordered.idx_is_valid(idx)) {
    return dims.ff_ordered.at(idx);
  } else {
    return std::nullopt;
  }
}

positive_int get_num_elements(TensorDims const &d) {
  return product(d.ff_ordered);
}

bool tensor_dims_is_broadcastable_to(TensorDims const &curr,
                                     TensorDims const &goal) {
  if (get_num_dims(curr) > get_num_dims(goal)) {
    return false;
  }

  std::vector<positive_int> curr_dims = vector_of(curr.ff_ordered);
  std::vector<positive_int> goal_dims = vector_of(goal.ff_ordered);

  for (auto const &[curr_dim, goal_dim] :
       zip(reversed(curr_dims), reversed(goal_dims))) {
    if (curr_dim != 1 && curr_dim != goal_dim) {
      return false;
    }
  }

  return true;
}

bool tensor_dims_contains_coord(TensorDims const &tensor_dims,
                                TensorDimsCoord const &coord) {
  ASSERT(coord.ff_ordered.size() == get_num_dims(tensor_dims));

  return all_are_true(zip_with(
      coord.ff_ordered,
      tensor_dims.ff_ordered,
      [](nonnegative_int const &coord_entry, positive_int const &dim_size) {
        return coord_entry < dim_size;
      }));
}

TensorDimsCoord get_broadcast_src_coord(TensorDims const &input_dims,
                                        TensorDims const &output_dims,
                                        TensorDimsCoord const &dst_coord) {
  ASSERT(tensor_dims_contains_coord(output_dims, dst_coord),
         output_dims,
         dst_coord);
  ASSERT(tensor_dims_is_broadcastable_to(input_dims, output_dims),
         input_dims,
         output_dims);

  relative_ff_dim_t trailing_start_idx =
      relative_ff_dim_t{-1 * get_num_dims(input_dims).unwrap_nonnegative()};

  FFOrdered<nonnegative_int> trailing_entries =
      slice(dst_coord.ff_ordered, trailing_start_idx);

  FFOrdered<positive_int> trailing_dims =
      slice(output_dims.ff_ordered, trailing_start_idx);

  TensorDimsCoord result = TensorDimsCoord{
      zip_with(trailing_entries,
               input_dims.ff_ordered,
               [](nonnegative_int const &coord_entry,
                  positive_int const &input_dim_size) {
                 if (input_dim_size == 1) {
                   return 0_n;
                 } else {
                   return coord_entry;
                 }
               }),
  };

  ASSERT(tensor_dims_contains_coord(input_dims, result),
         output_dims,
         dst_coord,
         input_dims,
         result);

  return result;
}

std::unordered_set<TensorDimsCoord> get_tensor_dims_coord_set(TensorDims const &tensor_dims) {  
  std::vector<std::vector<nonnegative_int>> per_dim_ranges = transform(
      vector_of(tensor_dims.ff_ordered),
      [](positive_int dim_size) -> std::vector<nonnegative_int> {
        return nonnegative_range(dim_size.nonnegative_int_from_positive_int());
      });

  std::unordered_set<std::vector<nonnegative_int>> raw_points =
      unordered_set_of(cartesian_product(per_dim_ranges));

  return transform(raw_points,
                   [](std::vector<nonnegative_int> const &raw_point) {
                     return TensorDimsCoord{ff_ordered_of(raw_point)};
                   });
}

std::unordered_set<ff_dim_t> get_ff_dim_t_set(TensorDims const &tensor_dims) {
  return unordered_set_of(get_idxs(tensor_dims.ff_ordered));
}

std::optional<TensorDims>
    get_broadcast_target_dims(std::unordered_set<TensorDims> const &dims) {
  for (TensorDims target_candidate : dims) {
    if (all_of(dims, [&](TensorDims const &d) {
          return tensor_dims_is_broadcastable_to(d, target_candidate);
        })) {
      return target_candidate;
    }
  }

  return std::nullopt;
}

TensorDims tensor_dims_drop_dims(TensorDims const &dims, std::function<bool(ff_dim_t)> const &should_drop_dim) {
  std::vector<positive_int> result;
  for (ff_dim_t idx : get_idxs(dims.ff_ordered)) {
    if (!should_drop_dim(idx)) {
      result.push_back(dims.ff_ordered.at(idx));
    }
  }

  return TensorDims{ff_ordered_of(result)};
}

TensorDims slice_tensor_dims(TensorDims const &dims,
                             relative_ff_dim_t const &start,
                             std::optional<relative_ff_dim_t> const &stop) {
  return TensorDims{
      slice(dims.ff_ordered, start, stop),
  };
}

TensorDims slice_tensor_dims(TensorDims const &dims,
                             ff_dim_t const &start,
                             std::optional<ff_dim_t> const &stop) {
  return TensorDims{
      slice(dims.ff_ordered, start, stop),
  };
}


} // namespace FlexFlow
