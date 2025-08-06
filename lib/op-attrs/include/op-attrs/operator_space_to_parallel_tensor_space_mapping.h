#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_MAPPING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_SPACE_TO_PARALLEL_TENSOR_SPACE_MAPPING_H

#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/parallel_tensor_dim_degrees.dtg.h"
#include "op-attrs/parallel_tensor_space_coordinate.dtg.h"
#include "op-attrs/parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include "op-attrs/num_ptensor_parallel_dims_t.h"

namespace FlexFlow {

OperatorSpaceToParallelTensorSpaceMapping
    get_identity_mapping(num_ptensor_parallel_dims_t num_parallel_dims);

OperatorSpaceToParallelTensorSpaceMapping
  operator_ptensor_space_mapping_from_composition(
    OperatorSpaceToParallelTensorSpaceMapping const &op_to_pt1_mapping,
    ParallelTensorSpaceMapping const &pt1_to_pt2_mapping);

ParallelTensorSpaceCoordinate
  ptensor_coord_for_task_space_coord(
    OperatorSpaceToParallelTensorSpaceMapping const &mapping,
    OperatorTaskSpace const &operator_task_space,
    ParallelTensorDimDegrees const &ptensor_dim_degrees,
    TaskSpaceCoordinate const &task_space_coord);

TaskSpaceCoordinate
  task_space_coord_for_ptensor_coord(
    OperatorSpaceToParallelTensorSpaceMapping const &mapping,
    ParallelTensorDimDegrees const &ptensor_dim_degrees,
    OperatorTaskSpace const &operator_task_space,
    ParallelTensorSpaceCoordinate const &tensor_space_coordinate);

} // namespace FlexFlow

#endif
