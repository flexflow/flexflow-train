#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_TASK_SPACE_TO_OPERATOR_TASK_SPACE_MAPPING_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_TASK_SPACE_TO_OPERATOR_TASK_SPACE_MAPPING_H

#include "op-attrs/operator_space_to_parallel_tensor_space_mapping.dtg.h"
#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/operator_task_space_to_operator_task_space_mapping.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"

namespace FlexFlow {

OperatorTaskSpace
  op_mapping_get_src_space(OperatorTaskSpaceToOperatorTaskSpaceMapping const &);

OperatorTaskSpace
  op_mapping_get_dst_space(OperatorTaskSpaceToOperatorTaskSpaceMapping const &);

bidict<TaskSpaceCoordinate, TaskSpaceCoordinate>
  op_to_op_get_coord_mapping(OperatorTaskSpaceToOperatorTaskSpaceMapping const &);

OperatorTaskSpaceToOperatorTaskSpaceMapping 
  op_to_op_mapping_from_composition_through_tensor(
    OperatorSpaceToParallelTensorSpaceMapping const &src_to_tensor_mapping,
    OperatorSpaceToParallelTensorSpaceMapping const &dst_to_tensor_mapping);

} // namespace FlexFlow

#endif
