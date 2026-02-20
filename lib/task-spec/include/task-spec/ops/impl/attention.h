#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_ATTENTION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_ATTENTION_H

#include "op-attrs/ops/attention.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_attention_init_task_impl();
TaskImplFunction get_attention_fwd_task_impl();
TaskImplFunction get_attention_bwd_task_impl();

} // namespace FlexFlow

#endif
