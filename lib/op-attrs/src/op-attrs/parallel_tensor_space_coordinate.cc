#include "op-attrs/parallel_tensor_space_coordinate.h"
#include "op-attrs/ff_ordered/ff_ordered_from_map.h"
#include "op-attrs/ff_ordered/ff_ordered_restrict_dims.h"
#include "op-attrs/ff_ordered/ff_ordered_without_dims.h"
#include "op-attrs/parallel_tensor_dim_degrees.h"
#include "op-attrs/parallel_tensor_dim_idx_t.h"
#include "utils/containers/contains_key.h"
#include "utils/containers/filtermap_keys.h"
#include "utils/containers/generate_map.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/orthotope/dim_coord.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/orthotope_bounded_coord.h"

namespace FlexFlow {

ParallelTensorSpaceCoordinate
    trivial_parallel_tensor_space_coordinate_for_num_tensor_dims(
        num_tensor_dims_t const &num_tensor_dims) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/0_n,
      /*discard_copy_component=*/0_n,
      /*shard_components=*/
      ff_ordered_of(
          transform(nonnegative_range(
                        num_tensor_dims.nonnegative_int_from_num_tensor_dims()),
                    [](nonnegative_int) -> nonnegative_int { return 0_n; })),
  };
}

num_ptensor_parallel_dims_t
    ptensor_coord_num_dims(ParallelTensorSpaceCoordinate const &c) {
  return num_ptensor_parallel_dims_t{
      2_n + num_elements(c.shard_components),
  };
}

num_ptensor_shard_dims_t
    ptensor_coord_num_shard_dims(ParallelTensorSpaceCoordinate const &c) {
  return num_ptensor_shard_dims_t{
      num_elements(c.shard_components),
  };
}

std::set<parallel_tensor_dim_idx_t> get_dim_idxs_in_ptensor_space_coord(
    ParallelTensorSpaceCoordinate const &coord) {

  std::set<parallel_tensor_dim_idx_t> result =
      dim_idxs_for_num_shard_dims(ptensor_coord_num_shard_dims(coord));
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

nonnegative_int &ptensor_coord_component_for_ptensor_dim_idx(
    ParallelTensorSpaceCoordinate &coord, parallel_tensor_dim_idx_t dim_idx) {
  if (dim_idx == sum_dim_idx()) {
    return coord.sum_component;
  } else if (dim_idx == discard_copy_dim_idx()) {
    return coord.discard_copy_component;
  } else {
    return coord.shard_components.at(dim_idx.require_shard_dim());
  }
}

ParallelTensorSpaceCoordinate
    parallel_tensor_space_coordinate_from_bounded_orthotope_components(
        BoundedComponent const &sum_component,
        BoundedComponent const &discard_copy_component,
        OrthotopeBoundedCoord const &shard_components) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/sum_component.get_component(),
      /*discard_copy_component=*/discard_copy_component.get_component(),
      /*shard_components=*/ff_ordered_of(shard_components.get_coord().raw),
  };
}

ParallelTensorDimDegrees smallest_parallel_tensor_dim_degrees_for_coord_set(
    std::set<ParallelTensorSpaceCoordinate> const &coord_set) {
  std::set<DimCoord<parallel_tensor_dim_idx_t>> dim_coord_set =
      transform(coord_set,
                [&](ParallelTensorSpaceCoordinate const &c)
                    -> DimCoord<parallel_tensor_dim_idx_t> {
                  return dim_coord_from_parallel_tensor_space_coord(c);
                });

  DimDomain<parallel_tensor_dim_idx_t> dim_domain =
      smallest_dim_domain_for_coord_set(dim_coord_set);

  return parallel_tensor_dim_degrees_from_dim_domain(dim_domain);
}

std::optional<ParallelTensorDimDegrees>
    strict_parallel_tensor_dim_degrees_for_coord_set(
        std::set<ParallelTensorSpaceCoordinate> const &coord_set) {
  std::set<DimCoord<parallel_tensor_dim_idx_t>> dim_coord_set =
      transform(coord_set,
                [&](ParallelTensorSpaceCoordinate const &c)
                    -> DimCoord<parallel_tensor_dim_idx_t> {
                  return dim_coord_from_parallel_tensor_space_coord(c);
                });

  std::optional<DimDomain<parallel_tensor_dim_idx_t>> dim_domain =
      strict_dim_domain_for_coord_set(dim_coord_set);

  if (dim_domain.has_value()) {
    return parallel_tensor_dim_degrees_from_dim_domain(dim_domain.value());
  } else {
    return std::nullopt;
  }
}

bool parallel_tensor_coord_set_is_orthotopic(
    std::set<ParallelTensorSpaceCoordinate> const &coord_set) {
  return strict_parallel_tensor_dim_degrees_for_coord_set(coord_set)
      .has_value();
}

OrthotopeBoundedCoord orthotope_bounded_coord_for_ptensor_dims(
    ParallelTensorDimDegrees const &degrees,
    ParallelTensorSpaceCoordinate const &coord,
    std::set<parallel_tensor_dim_idx_t> const &desired_dims) {
  return make_orthotope_bounded_coord_from_components(transform(
      sorted_by(desired_dims, get_parallel_tensor_dim_ordering().lt),
      [&](parallel_tensor_dim_idx_t dim_idx) -> BoundedComponent {
        return bounded_component_for_ptensor_dim(degrees, coord, dim_idx);
      }));
}

BoundedComponent bounded_component_for_ptensor_dim(
    ParallelTensorDimDegrees const &degrees,
    ParallelTensorSpaceCoordinate const &coord,
    parallel_tensor_dim_idx_t const &desired_dim) {
  ASSERT(parallel_tensor_space_contains_coord(degrees, coord));

  return BoundedComponent{
      /*component=*/ptensor_coord_component_for_ptensor_dim_idx(coord,
                                                                desired_dim),
      /*bound=*/get_degree_for_parallel_tensor_dim_idx(degrees, desired_dim),
  };
}

ParallelTensorSpaceCoordinate parallel_tensor_space_coord_from_map(
    std::map<parallel_tensor_dim_idx_t, nonnegative_int> const &m) {
  ASSERT(contains_key(m, sum_dim_idx()));
  ASSERT(contains_key(m, discard_copy_dim_idx()));

  std::map<ff_dim_t, nonnegative_int> shard_map =
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

DimCoord<parallel_tensor_dim_idx_t> dim_coord_from_parallel_tensor_space_coord(
    ParallelTensorSpaceCoordinate const &coord) {

  return DimCoord<parallel_tensor_dim_idx_t>{
      generate_map(get_dim_idxs_in_ptensor_space_coord(coord),
                   [&](parallel_tensor_dim_idx_t idx) {
                     return ptensor_coord_component_for_ptensor_dim_idx(coord,
                                                                        idx);
                   }),
  };
}

ParallelTensorSpaceCoordinate parallel_tensor_space_coordinate_restrict_dims(
    ParallelTensorSpaceCoordinate const &coord,
    std::set<ff_dim_t> const &dim_set) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/coord.sum_component,
      /*discard_copy_component=*/coord.discard_copy_component,
      /*shard_components=*/
      ff_ordered_restrict_dims(coord.shard_components, dim_set),
  };
}

ParallelTensorSpaceCoordinate parallel_tensor_space_coordinate_without_dims(
    ParallelTensorSpaceCoordinate const &coord,
    std::set<ff_dim_t> const &dim_set) {
  return ParallelTensorSpaceCoordinate{
      /*sum_component=*/coord.sum_component,
      /*discard_copy_component=*/coord.discard_copy_component,
      /*shard_components=*/
      ff_ordered_without_dims(coord.shard_components, dim_set),
  };
}

} // namespace FlexFlow
