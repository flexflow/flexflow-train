#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_LAYER_NORM_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_LAYER_NORM_H

#include "op-attrs/ops/layer_norm_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_layer_norm_init_task_impl();
TaskImplFunction get_layer_norm_fwd_task_impl();
TaskImplFunction get_layer_norm_bwd_task_impl();

} // namespace FlexFlow

#endif
