#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_DEGREES_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIM_DEGREES_H

#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"

namespace FlexFlow {

std::set<parallel_tensor_dim_idx_t> get_nontrivial_parallel_tensor_dim_indices(ParallelTensorDimDegrees const &);

std::unordered_map<parallel_tensor_dim_idx_t, int>
  get_parallel_tensor_degree_map(ParallelTensorDimDegrees const &);

std::unordered_set<ParallelTensorSpaceCoordinate>
  get_parallel_tensor_space_coordinates(ParallelTensorDimDegrees const &);

} // namespace FlexFlow

#endif
