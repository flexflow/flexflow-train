#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_RESHAPE_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_RESHAPE_H

#include "op-attrs/ops/reshape_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(ReshapeAttrs const &);

TaskImplFunction get_reshape_fwd_task_impl();
TaskImplFunction get_reshape_bwd_task_impl();

OpTaskSignature get_reshape_fwd_signature();
OpTaskSignature get_reshape_bwd_signature();

OpTaskInvocation forward(ReshapeAttrs const &);
OpTaskInvocation backward(ReshapeAttrs const &);

} // namespace FlexFlow

#endif
