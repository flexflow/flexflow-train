#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_DEGREES_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_DEGREES_H

#include "op-attrs/num_ptensor_shard_dims_t.dtg.h"
#include "op-attrs/num_tensor_dims_t.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/minimal_dim_domain.dtg.h"

namespace FlexFlow {

num_ptensor_shard_dims_t get_ptensor_dim_degrees_num_shard_dims(ParallelTensorDimDegrees const &);
num_tensor_dims_t get_ptensor_dim_degrees_num_tensor_dims(ParallelTensorDimDegrees const &);

std::unordered_set<parallel_tensor_dim_idx_t>
  get_parallel_tensor_dim_indices(ParallelTensorDimDegrees const &);

std::set<parallel_tensor_dim_idx_t> get_nontrivial_parallel_tensor_dim_indices(
    ParallelTensorDimDegrees const &);

positive_int get_degree_for_parallel_tensor_dim_idx(ParallelTensorDimDegrees const &,
                                                    parallel_tensor_dim_idx_t const &);

std::unordered_map<parallel_tensor_dim_idx_t, positive_int>
    get_parallel_tensor_degree_map(ParallelTensorDimDegrees const &);

std::unordered_set<ParallelTensorSpaceCoordinate>
    get_parallel_tensor_space_coordinates(ParallelTensorDimDegrees const &);

DimDomain<parallel_tensor_dim_idx_t>
  dim_domain_from_parallel_tensor_dim_degrees(ParallelTensorDimDegrees const &);

MinimalDimDomain<parallel_tensor_dim_idx_t>
  minimal_dim_domain_from_parallel_tensor_dim_degrees(ParallelTensorDimDegrees const &);


} // namespace FlexFlow

#endif
