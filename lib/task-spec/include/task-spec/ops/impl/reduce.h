#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_REDUCE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_REDUCE_H

#include "op-attrs/ops/reduce_attrs.dtg.h"
#include "task-spec/ops/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_reduce_init_task_impl();
TaskImplFunction get_reduce_fwd_task_impl();
TaskImplFunction get_reduce_bwd_task_impl();

OpTaskInvocation init(ReduceAttrs const &);
OpTaskInvocation forward(ReduceAttrs const &);
OpTaskInvocation backward(ReduceAttrs const &);

} // namespace FlexFlow

#endif
