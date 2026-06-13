#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_TRAINING_OPERATION_ATTRS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_TRAINING_OPERATION_ATTRS_H

#include "task-spec/dynamic_graph/training_op_type.dtg.h"
#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"

namespace FlexFlow {

TrainingOpType training_op_attrs_get_op_type(TrainingOperationAttrs const &);

} // namespace FlexFlow

#endif
