#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_TASK_TYPE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_TASK_TYPE_H

#include "task-spec/dynamic_graph/dynamic_task_type.dtg.h"
#include "task-spec/dynamic_graph/dynamic_tensor_role.dtg.h"

namespace FlexFlow {

DynamicTaskType dynamic_task_type_from_tensor_role_for_copy(DynamicTensorRole);

} // namespace FlexFlow

#endif
