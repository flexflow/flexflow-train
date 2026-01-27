#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_DROPOUT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_DROPOUT_H

#include "op-attrs/ops/dropout_attrs.dtg.h"
#include "task-spec/task_id_t.dtg.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_dropout_init_task_impl();
TaskImplFunction get_dropout_fwd_task_impl();
TaskImplFunction get_dropout_bwd_task_impl();

} // namespace FlexFlow

#endif
