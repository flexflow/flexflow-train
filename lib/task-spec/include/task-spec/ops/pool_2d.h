#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_POOL_2D_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_POOL_2D_H

#include "op-attrs/ops/pool_2d_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(Pool2DAttrs const &);

TaskImplFunction get_pool_2d_init_task_impl();
TaskImplFunction get_pool_2d_fwd_task_impl();
TaskImplFunction get_pool_2d_bwd_task_impl();

OpTaskSignature get_pool_2d_init_signature();
OpTaskSignature get_pool_2d_fwd_signature();
OpTaskSignature get_pool_2d_bwd_signature();

OpTaskInvocation init(Pool2DAttrs const &);
OpTaskInvocation forward(Pool2DAttrs const &);
OpTaskInvocation backward(Pool2DAttrs const &);

} // namespace FlexFlow

#endif
