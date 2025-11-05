#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_LINEAR_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_LINEAR_H

#include "op-attrs/ops/linear_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(LinearAttrs const &);

OpTaskInvocation init(LinearAttrs const &);
OpTaskInvocation forward(LinearAttrs const &);
OpTaskInvocation backward(LinearAttrs const &);

TaskImplFunction get_linear_init_task_impl();
TaskImplFunction get_linear_fwd_task_impl();
TaskImplFunction get_linear_bwd_task_impl();

OpTaskSignature get_linear_init_signature();
OpTaskSignature get_linear_fwd_signature();
OpTaskSignature get_linear_bwd_signature();

} // namespace FlexFlow

#endif
