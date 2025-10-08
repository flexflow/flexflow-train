#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_INVOCATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_INVOCATION_H

#include "task-spec/runtime_task_invocation.dtg.h"

namespace FlexFlow {

bool is_invocation_valid(RuntimeTaskSignature const &sig, RuntimeTaskInvocation const &inv);

}

#endif
