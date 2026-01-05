#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_CONCAT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_CONCAT_H

#include "op-attrs/ops/concat_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_concat_fwd_task_impl();
TaskImplFunction get_concat_bwd_task_impl();

} // namespace FlexFlow

#endif
