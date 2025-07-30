#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "op-attrs/ff_dim_t.h"
#include "utils/containers/set_of.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

parallel_tensor_dim_idx_t sum_dim_idx() {
  return parallel_tensor_dim_idx_t{ReplicaType::SUM};
}

parallel_tensor_dim_idx_t discard_copy_dim_idx() {
  return parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY};
}

parallel_tensor_dim_idx_t shard_dim_idx(ff_dim_t idx) {
  return parallel_tensor_dim_idx_t{idx};
}

bool operator<(parallel_tensor_dim_idx_t lhs, parallel_tensor_dim_idx_t rhs) {
  if (lhs.is_shard_dim() && rhs.is_shard_dim()) {
    return lhs.require_shard_dim() < rhs.require_shard_dim(); 
  } else if (lhs.is_shard_dim() && !rhs.is_shard_dim()) {
    return true;
  } else if (!lhs.is_shard_dim() && rhs.is_shard_dim()) {
    return false;
  } else {
    return lhs.require_replica_dim() < rhs.require_replica_dim();
  }
}

std::set<parallel_tensor_dim_idx_t>
    dim_idxs_for_num_shard_dims(nonnegative_int num_shard_dims) {
  std::set<parallel_tensor_dim_idx_t> result =
      transform(set_of(ff_dim_range(num_shard_dims)), shard_dim_idx);
  result.insert(sum_dim_idx());
  result.insert(discard_copy_dim_idx());

  return result;
}

} // namespace FlexFlow
