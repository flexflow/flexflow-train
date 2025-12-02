#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_POOL_2D_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_POOL_2D_H

#include "op-attrs/ops/pool_2d_attrs.dtg.h"
#include "task-spec/ops/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_pool_2d_init_task_impl();
TaskImplFunction get_pool_2d_fwd_task_impl();
TaskImplFunction get_pool_2d_bwd_task_impl();

OpTaskInvocation init(Pool2DAttrs const &);
OpTaskInvocation forward(Pool2DAttrs const &);
OpTaskInvocation backward(Pool2DAttrs const &);

} // namespace FlexFlow

#endif
