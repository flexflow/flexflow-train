#ifndef _FLEXFLOW_TRANSPOSE_H_
#define _FLEXFLOW_TRANSPOSE_H_

#include "local-execution/task_impl_function.dtg.h"
#include "op-attrs/ops/transpose_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(TransposeAttrs const &);

TaskImplFunction get_transpose_fwd_task_impl();
TaskImplFunction get_transpose_bwd_task_impl();

OpTaskSignature get_transpose_fwd_signature();
OpTaskSignature get_transpose_bwd_signature();

OpTaskInvocation forward(TransposeAttrs const &);
OpTaskInvocation backward(TransposeAttrs const &);

} // namespace FlexFlow

#endif
