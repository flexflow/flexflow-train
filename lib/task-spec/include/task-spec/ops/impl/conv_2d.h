#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_CONV_2D_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_CONV_2D_H

#include "op-attrs/ops/conv_2d_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_conv_2d_init_task_impl();
TaskImplFunction get_conv_2d_fwd_task_impl();
TaskImplFunction get_conv_2d_bwd_task_impl();

} // namespace FlexFlow

#endif
