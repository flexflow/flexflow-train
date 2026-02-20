#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_LINEAR_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_LINEAR_H

#include "op-attrs/ops/linear_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_linear_init_task_impl();
TaskImplFunction get_linear_fwd_task_impl();
TaskImplFunction get_linear_bwd_task_impl();

} // namespace FlexFlow

#endif
