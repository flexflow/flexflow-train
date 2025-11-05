#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_GATHER_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_GATHER_H

#include "op-attrs/ops/gather_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(GatherAttrs const &);

TaskImplFunction get_gather_init_task_impl();
TaskImplFunction get_gather_fwd_task_impl();
TaskImplFunction get_gather_bwd_task_impl();

OpTaskSignature get_gather_init_signature();
OpTaskSignature get_gather_fwd_signature();
OpTaskSignature get_gather_bwd_signature();

OpTaskInvocation init(GatherAttrs const &);
OpTaskInvocation forward(GatherAttrs const &);
OpTaskInvocation backward(GatherAttrs const &);

} // namespace FlexFlow

#endif
