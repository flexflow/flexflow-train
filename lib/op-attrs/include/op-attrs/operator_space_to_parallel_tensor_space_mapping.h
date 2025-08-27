#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_MAPPING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_MAPPING_H

#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "op-attrs/parallel_tensor_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "op-attrs/num_ptensor_parallel_dims_t.h"

namespace FlexFlow {

OperatorTaskSpace
  get_operator_task_space_for_mapping(OperatorSpaceToParallelTensorSpaceMapping const &);

ParallelTensorDimDegrees
  get_parallel_tensor_space_for_mapping(OperatorSpaceToParallelTensorSpaceMapping const &);

OperatorSpaceToParallelTensorSpaceMapping
    get_identity_mapping(
      OperatorTaskSpace const &operator_task_space,
      ParallelTensorDimDegrees const &parallel_tensor_dim_degrees);

OperatorSpaceToParallelTensorSpaceMapping
    operator_ptensor_space_mapping_from_projection(
      DimProjection<operator_task_space_dim_idx_t, parallel_tensor_dim_idx_t> const &projection,
      OperatorTaskSpace const &op_task_space,
      ParallelTensorDimDegrees const &parallel_tensor_dim_degrees);

OperatorSpaceToParallelTensorSpaceMapping
  operator_ptensor_space_mapping_from_composition(
    OperatorSpaceToParallelTensorSpaceMapping const &op_to_pt1_mapping,
    ParallelTensorSpaceToParallelTensorSpaceMapping const &pt1_to_pt2_mapping);

ParallelTensorSpaceCoordinate
  ptensor_coord_for_task_space_coord(
    OperatorSpaceToParallelTensorSpaceMapping const &mapping,
    TaskSpaceCoordinate const &task_space_coord);

TaskSpaceCoordinate
  task_space_coord_for_ptensor_coord(
    OperatorSpaceToParallelTensorSpaceMapping const &mapping,
    ParallelTensorSpaceCoordinate const &tensor_space_coordinate);

} // namespace FlexFlow

#endif
