#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/ff_ordered/ff_ordered_from_map.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/containers/filtermap_keys.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

std::unordered_set<parallel_tensor_dim_idx_t>
  get_dim_idxs_in_ptensor_space_coord(ParallelTensorSpaceCoordinate const &coord) {
  
  std::unordered_set<parallel_tensor_dim_idx_t> result = 
    unordered_set_of(dim_idxs_for_num_shard_dims(num_elements(coord.shard_components)));
  result.insert(sum_dim_idx());
  result.insert(discard_copy_dim_idx());
  return result;
}


nonnegative_int ptensor_coord_component_for_ptensor_dim_idx(
    ParallelTensorSpaceCoordinate const &coord,
    parallel_tensor_dim_idx_t dim_idx) {
  if (dim_idx == sum_dim_idx()) {
    return coord.sum_component;
  } else if (dim_idx == discard_copy_dim_idx()) {
    return coord.discard_copy_component;
  } else {
    return coord.shard_components.at(dim_idx.require_shard_dim());
  }
}

ParallelTensorSpaceCoordinate parallel_tensor_space_coord_from_map(
    std::unordered_map<parallel_tensor_dim_idx_t, nonnegative_int> const &m) {

  std::unordered_map<ff_dim_t, nonnegative_int> shard_map =
      filtermap_keys(m, [](parallel_tensor_dim_idx_t const &d) {
        return d.try_require_shard_dim();
      });

  return ParallelTensorSpaceCoordinate{
      /*sum_idx=*/m.at(parallel_tensor_dim_idx_t{ReplicaType::SUM}),
      /*discard_copy_idx=*/
      m.at(parallel_tensor_dim_idx_t{ReplicaType::DISCARD_COPY}),
      /*shard_idxs=*/ff_ordered_from_map(shard_map),
  };
}

ParallelTensorSpaceCoordinate parallel_tensor_space_coord_from_dim_coord(
    DimCoord<parallel_tensor_dim_idx_t> const &dim_coord) {
  return parallel_tensor_space_coord_from_map(dim_coord.raw);
}

DimCoord<parallel_tensor_dim_idx_t> 
  dim_coord_from_parallel_tensor_space_coord(
    ParallelTensorSpaceCoordinate const &coord) {

  return DimCoord<parallel_tensor_dim_idx_t>{
    generate_map(get_dim_idxs_in_ptensor_space_coord(coord),
                 [&](parallel_tensor_dim_idx_t idx) {
                   return ptensor_coord_component_for_ptensor_dim_idx(coord, idx);
                 }),
  };
}

} // namespace FlexFlow
