#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_EMBEDDING_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_EMBEDDING_H

#include "op-attrs/ops/embedding_attrs.dtg.h"
#include "task-spec/ops/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_embedding_fwd_task_impl();
TaskImplFunction get_embedding_bwd_task_impl();

OpTaskSignature get_embedding_fwd_signature();
OpTaskSignature get_embedding_bwd_signature();

OpTaskInvocation forward(EmbeddingAttrs const &);
OpTaskInvocation backward(EmbeddingAttrs const &);

} // namespace FlexFlow

#endif
