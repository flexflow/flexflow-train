#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_ATTENTION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_ATTENTION_H

#include "op-attrs/ops/attention.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(MultiHeadAttentionAttrs const &);

TaskImplFunction get_attention_init_task_impl();
TaskImplFunction get_attention_fwd_task_impl();
TaskImplFunction get_attention_bwd_task_impl();

OpTaskSignature get_attention_init_signature();
OpTaskSignature get_attention_fwd_signature();
OpTaskSignature get_attention_bwd_signature();

OpTaskInvocation init(MultiHeadAttentionAttrs const &);
OpTaskInvocation forward(MultiHeadAttentionAttrs const &);
OpTaskInvocation backward(MultiHeadAttentionAttrs const &);

} // namespace FlexFlow

#endif
