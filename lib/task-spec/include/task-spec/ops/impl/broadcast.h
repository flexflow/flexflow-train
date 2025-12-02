#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_BROADCAST_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_BROADCAST_H

#include "op-attrs/ops/broadcast_attrs.dtg.h"
#include "task-spec/ops/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

OpTaskInvocation forward(BroadcastAttrs const &);
OpTaskInvocation backward(BroadcastAttrs const &);

TaskImplFunction get_broadcast_fwd_task_impl();
TaskImplFunction get_broadcast_bwd_task_impl();

} // namespace FlexFlow

#endif
