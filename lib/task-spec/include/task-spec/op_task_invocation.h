#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_INVOCATION_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TASK_INVOCATION_H

#include "task-spec/op_task_invocation.dtg.h"
#include "task-spec/op_task_signature.h"

namespace FlexFlow {

bool is_invocation_valid(OpTaskSignature const &sig,
                         OpTaskInvocation const &inv);

} // namespace FlexFlow

#endif
