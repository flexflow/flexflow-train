#include "op-attrs/tensor_dims.h"
#include "op-attrs/ff_ordered/slice.h"
#include "op-attrs/ff_ordered/zip.h"
#include "op-attrs/replica_parallel_dim_set.h"
#include "op-attrs/shard_parallel_dim.dtg.h"
#include "utils/containers/all_of.h"
#include "utils/containers/product.h"
#include "utils/containers/reversed.h"
#include "utils/containers/transform.h"
#include "utils/containers/vector_of.h"
#include "utils/containers/zip.h"
#include "utils/integer_conversions.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

FFOrdered<nonnegative_int> const &ff_ordered(TensorDims const &dims) {
  return dims.ff_ordered;
}

nonnegative_int num_dims(TensorDims const &dims) {
  return num_elements(dims.ff_ordered);
}

nonnegative_int dim_at_idx(TensorDims const &dims, relative_ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

nonnegative_int &dim_at_idx(TensorDims &dims, relative_ff_dim_t idx) {
  return dims.ff_ordered.at(idx);
}

nonnegative_int get_num_elements(TensorDims const &d) {
  return product(d.ff_ordered);
}

bool tensor_dims_is_broadcastable_to(TensorDims const &curr,
                                     TensorDims const &goal) {
  if (num_dims(curr) > num_dims(goal)) {
    return false;
  }

  std::vector<nonnegative_int> curr_dims = vector_of(curr.ff_ordered);
  std::vector<nonnegative_int> goal_dims = vector_of(goal.ff_ordered);

  for (auto const &[curr_dim, goal_dim] :
       zip(reversed(curr_dims), reversed(goal_dims))) {
    if (curr_dim != 1 && curr_dim != goal_dim) {
      return false;
    }
  }

  return true;
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
