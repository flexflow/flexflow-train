#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_BATCH_NORM_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_BATCH_NORM_H

#include "op-attrs/ops/batch_norm_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_batch_norm_init_task_impl();
TaskImplFunction get_batch_norm_fwd_task_impl();
TaskImplFunction get_batch_norm_bwd_task_impl();

} // namespace FlexFlow

#endif
