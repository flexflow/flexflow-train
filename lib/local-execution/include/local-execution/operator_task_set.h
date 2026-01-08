#ifndef _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_OPERATOR_TASK_SET_H
#define _FLEXFLOW_LIB_LOCAL_EXECUTION_INCLUDE_LOCAL_EXECUTION_OPERATOR_TASK_SET_H

#include "local-execution/operator_task_set.dtg.h"
#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "task-spec/ops/op_task_type.dtg.h"
#include "utils/bidict/bidict.h"

namespace FlexFlow {

bidict<OpTaskType, task_id_with_noop_default_t>
    get_map_from_task_type_to_task(OperatorTaskSet const &);
std::unordered_set<task_id_with_noop_default_t>
    get_all_tasks_in_task_set(OperatorTaskSet const &);

task_id_with_noop_default_t
    get_task_for_task_type(OperatorTaskSet const &op_task_set,
                           OpTaskType task_type);

OperatorTaskSet
    get_task_set_for_operator(ComputationGraphOpAttrs const &op_attrs);

} // namespace FlexFlow

#endif
