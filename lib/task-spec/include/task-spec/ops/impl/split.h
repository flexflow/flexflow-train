#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_SPLIT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_SPLIT_H

#include "op-attrs/ops/split_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(SplitAttrs const &);

TaskImplFunction get_split_fwd_task_impl();
TaskImplFunction get_split_bwd_task_impl();

OpTaskSignature get_split_fwd_signature();
OpTaskSignature get_split_bwd_signature();

OpTaskInvocation forward(SplitAttrs const &);
OpTaskInvocation backward(SplitAttrs const &);

} // namespace FlexFlow

#endif
