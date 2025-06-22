#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_TOPK_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_TOPK_H

#include "op-attrs/ops/topk_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(TopKAttrs const &);

TaskImplFunction get_topk_fwd_task_impl();
TaskImplFunction get_topk_bwd_task_impl();

OpTaskSignature get_topk_fwd_signature();
OpTaskSignature get_topk_bwd_signature();

OpTaskInvocation forward(TopKAttrs const &);
OpTaskInvocation backward(TopKAttrs const &);

} // namespace FlexFlow

#endif
