#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_TASK_SPACE_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_OPERATOR_TASK_SPACE_H

#include "op-attrs/operator_task_space.dtg.h"
#include "op-attrs/task_space_coordinate.dtg.h"
#include <unordered_set>

namespace FlexFlow {

std::unordered_set<TaskSpaceCoordinate>
    get_task_space_coordinates(OperatorTaskSpace const &task);

TaskSpaceCoordinate
    get_task_space_maximum_coordinate(OperatorTaskSpace const &task);

nonnegative_int num_dims(OperatorTaskSpace const &task);
positive_int num_tasks(OperatorTaskSpace const &task);

} // namespace FlexFlow

#endif
