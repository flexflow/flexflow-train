#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_REVERSE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_REVERSE_H

#include "op-attrs/ops/reverse_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_reverse_fwd_task_impl();
TaskImplFunction get_reverse_bwd_task_impl();

} // namespace FlexFlow

#endif
