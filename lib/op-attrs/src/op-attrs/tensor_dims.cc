#include "op-attrs/tensor_dims.h"
#include "op-attrs/ff_ordered/slice.h"
#include "op-attrs/ff_ordered/zip.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.dtg.h"
#include "utils/containers/all_are_true.h"
#include "utils/containers/all_of.h"
#include "utils/containers/product.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/zip.h"
#include "op-attrs/ff_ordered/zip_with.h"
#include "utils/integer_conversions.h"
#include "utils/nonnegative_int/num_elements.h"
#include "op-attrs/ff_ordered/slice.h"

namespace FlexFlow {

FFOrdered<positive_int> const &ff_ordered(TensorDims const &dims) {
  return dims.ff_ordered;
}

nonnegative_int num_dims(TensorDims const &dims) {
  return num_elements(dims.ff_ordered);
}

positive_int dim_at_idx(TensorDims const &dims, relative_ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

positive_int &dim_at_idx(TensorDims &dims, relative_ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

positive_int get_num_elements(TensorDims const &d) {
  return product(d.ff_ordered);
}

bool tensor_dims_is_broadcastable_to(TensorDims const &curr,
                                     TensorDims const &goal) {
  if (num_dims(curr) > num_dims(goal)) {
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

bool tensor_dims_contains_coord(TensorDims const &tensor_dims, TensorDimsCoord const &coord) {
  ASSERT(coord.ff_ordered.size() == num_dims(tensor_dims));

  return all_are_true(
    zip_with(
      coord.ff_ordered, 
      tensor_dims.ff_ordered, 
      [](nonnegative_int const &coord_entry, positive_int const &dim_size) { 
        return coord_entry < dim_size;
      }));
}

TensorDimsCoord get_broadcast_src_coord(TensorDims const &input_dims,
                                        TensorDims const &output_dims,
                                        TensorDimsCoord const &dst_coord) {
  ASSERT(tensor_dims_contains_coord(output_dims, dst_coord));   
  ASSERT(tensor_dims_is_broadcastable_to(input_dims, output_dims));

  relative_ff_dim_t trailing_start_idx = relative_ff_dim_t{-1 * num_dims(input_dims).unwrap_nonnegative()};

  FFOrdered<nonnegative_int> trailing_entries = 
    slice(dst_coord.ff_ordered, trailing_start_idx);

  FFOrdered<positive_int> trailing_dims = slice(output_dims.ff_ordered, trailing_start_idx);

  TensorDimsCoord result = TensorDimsCoord{
    zip_with(
      trailing_entries,
      trailing_dims,
      [](nonnegative_int const &coord_entry, positive_int const &dim_size) {
        if (dim_size == 1) {
          return 0_n;
        } else {
          return coord_entry;
        }
      }),
  };

  ASSERT(tensor_dims_contains_coord(input_dims, result));

  return result;
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

TensorDims slice_tensor_dims(TensorDims const &dims,
                             relative_ff_dim_t const &start,
                             std::optional<relative_ff_dim_t> const &stop) {
  return TensorDims{
      slice(dims.ff_ordered, start, stop),
  };
}

} // namespace FlexFlow
