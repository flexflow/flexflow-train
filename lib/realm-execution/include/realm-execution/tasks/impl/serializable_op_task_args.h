#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_OP_TASK_ARGS_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_OP_TASK_ARGS_H

#include "realm-execution/tasks/impl/op_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_op_task_args.dtg.h"

namespace FlexFlow {

SerializableOpTaskArgs op_task_args_to_serializable(OpTaskArgs const &);
OpTaskArgs op_task_args_from_serializable(SerializableOpTaskArgs const &);

} // namespace FlexFlow

#endif
