#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SPACE_MAPPING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_SPACE_MAPPING_H

#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_dim_idx_t.dtg.h"
#include "op-attrs/parallel_tensor_space_mapping.dtg.h"
#include "utils/orthotope/dim_projection.dtg.h"

namespace FlexFlow {

ParallelTensorSpaceMapping
  parallel_tensor_space_mapping_from_projection(
    DimProjection<parallel_tensor_dim_idx_t, parallel_tensor_dim_idx_t> const &projection, 
    ParallelTensorDimDegrees const &l_degrees,
    ParallelTensorDimDegrees const &r_degrees);


} // namespace FlexFlow

#endif
