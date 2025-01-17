#include "op-attrs/parallel_tensor_dim_idx_t.h"

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

} // namespace FlexFlow
