#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_TRAINING_OPERATION_ATTRS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_TRAINING_OPERATION_ATTRS_H

#include "task-spec/dynamic_graph/training_operation_attrs.dtg.h"
#include "op-attrs/operator_type.dtg.h"

namespace FlexFlow {

bool training_op_attrs_has_op_type(TrainingOperationAttrs const &, OperatorType);

} // namespace FlexFlow

#endif
