#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_CONCAT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_CONCAT_H

#include "op-attrs/ops/concat_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(ConcatAttrs const &);

TaskImplFunction get_concat_fwd_task_impl();
TaskImplFunction get_concat_bwd_task_impl();

OpTaskSignature get_concat_fwd_signature();
OpTaskSignature get_concat_bwd_signature();

OpTaskInvocation forward(ConcatAttrs const &);
OpTaskInvocation backward(ConcatAttrs const &);

} // namespace FlexFlow

#endif
