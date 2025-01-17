#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SPACE_COORDINATE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SPACE_COORDINATE_H

#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"

namespace FlexFlow {

ParallelTensorSpaceCoordinate 
  parallel_tensor_space_coord_from_map(std::unordered_map<parallel_tensor_dim_idx_t, int> const &);

} // namespace FlexFlow

#endif
