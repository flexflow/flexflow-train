#ifndef _FLEXFLOW_COMBINE_H
#define _FLEXFLOW_COMBINE_H

#include "task-spec/task_impl_function.dtg.h"
#include "op-attrs/ops/combine_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(CombineAttrs const &);

TaskImplFunction get_combine_fwd_task_impl();
TaskImplFunction get_combine_bwd_task_impl();

OpTaskSignature get_combine_fwd_signature();
OpTaskSignature get_combine_bwd_signature();

OpTaskInvocation forward(CombineAttrs const &);
OpTaskInvocation backward(CombineAttrs const &);

} // namespace FlexFlow

#endif
