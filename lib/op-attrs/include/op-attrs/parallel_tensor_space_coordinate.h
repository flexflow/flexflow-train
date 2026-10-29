#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SPACE_COORDINATE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SPACE_COORDINATE_H

#include "op-attrs/num_ptensor_parallel_dims_t.h"
#include "op-attrs/num_ptensor_shard_dims_t.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "utils/orthotope/bounded_component.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/orthotope_bounded_coord.h"

namespace FlexFlow {

num_ptensor_parallel_dims_t
    ptensor_coord_num_dims(ParallelTensorSpaceCoordinate const &);
num_ptensor_shard_dims_t
    ptensor_coord_num_shard_dims(ParallelTensorSpaceCoordinate const &);

std::set<parallel_tensor_dim_idx_t>
    get_dim_idxs_in_ptensor_space_coord(ParallelTensorSpaceCoordinate const &);

nonnegative_int ptensor_coord_component_for_ptensor_dim_idx(
    ParallelTensorSpaceCoordinate const &, parallel_tensor_dim_idx_t);

nonnegative_int &
    ptensor_coord_component_for_ptensor_dim_idx(ParallelTensorSpaceCoordinate &,
                                                parallel_tensor_dim_idx_t);

ParallelTensorSpaceCoordinate
    parallel_tensor_space_coordinate_from_bounded_orthotope_components(
        BoundedComponent const &sum_component,
        BoundedComponent const &discard_copy_component,
        OrthotopeBoundedCoord const &shard_components);

ParallelTensorDimDegrees smallest_parallel_tensor_dim_degrees_for_coord_set(
    std::set<ParallelTensorSpaceCoordinate> const &);

std::optional<ParallelTensorDimDegrees>
    strict_parallel_tensor_dim_degrees_for_coord_set(
        std::set<ParallelTensorSpaceCoordinate> const &);

bool parallel_tensor_coord_set_is_orthotopic(
    std::set<ParallelTensorSpaceCoordinate> const &);

OrthotopeBoundedCoord orthotope_bounded_coord_for_ptensor_dims(
    ParallelTensorDimDegrees const &,
    ParallelTensorSpaceCoordinate const &,
    std::set<parallel_tensor_dim_idx_t> const &);

BoundedComponent
    bounded_component_for_ptensor_dim(ParallelTensorDimDegrees const &,
                                      ParallelTensorSpaceCoordinate const &,
                                      parallel_tensor_dim_idx_t const &);

ParallelTensorSpaceCoordinate parallel_tensor_space_coord_from_map(
    std::map<parallel_tensor_dim_idx_t, nonnegative_int> const &);

ParallelTensorSpaceCoordinate parallel_tensor_space_coord_from_dim_coord(
    DimCoord<parallel_tensor_dim_idx_t> const &);

DimCoord<parallel_tensor_dim_idx_t> dim_coord_from_parallel_tensor_space_coord(
    ParallelTensorSpaceCoordinate const &);

} // namespace FlexFlow

#endif
