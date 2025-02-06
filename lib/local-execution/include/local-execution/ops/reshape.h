#ifndef _FLEXFLOW_RESHAPE_H
#define _FLEXFLOW_RESHAPE_H

#include "local-execution/task_impl_function.dtg.h"
#include "op-attrs/ops/reshape_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(ReshapeAttrs const &);

TaskImplFunction get_reshape_init_task_impl();
TaskImplFunction get_reshape_fwd_task_impl();
TaskImplFunction get_reshape_bwd_task_impl();

OpTaskSignature get_reshape_init_signature();
OpTaskSignature get_reshape_fwd_signature();
OpTaskSignature get_reshape_bwd_signature();

OpTaskInvocation init(ReshapeAttrs const &);
OpTaskInvocation forward(ReshapeAttrs const &);
OpTaskInvocation backward(ReshapeAttrs const &);

} // namespace FlexFlow

#endif
