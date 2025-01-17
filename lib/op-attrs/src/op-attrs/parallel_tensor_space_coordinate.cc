#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/dim_ordered/ff_ordered_from_map.h"
#include "utils/containers/filtermap_keys.h"

namespace FlexFlow {

ParallelTensorSpaceCoordinate 
  parallel_tensor_space_coord_from_map(std::unordered_map<parallel_tensor_dim_idx_t, int> const &m) {

  std::unordered_map<ff_dim_t, int> shard_map = filtermap_keys
    (m, [](parallel_tensor_dim_idx_t const &d) { return d.try_require_shard_dim(); });

  return ParallelTensorSpaceCoordinate{
    /*sum_idx=*/m.at(parallel_tensor_dim_idx_t{ReplicaType::SUM}),
    /*discard_copy_idx=*/m.at(parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY}),
    /*shard_idxs=*/ff_ordered_from_map(shard_map),
  };
}

} // namespace FlexFlow
