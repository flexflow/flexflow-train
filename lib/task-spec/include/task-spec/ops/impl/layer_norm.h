#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_LAYER_NORM_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_LAYER_NORM_H

#include "op-attrs/ops/layer_norm_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

std::unordered_set<task_id_t> get_task_ids(LayerNormAttrs const &);

TaskImplFunction get_layer_norm_init_task_impl();
TaskImplFunction get_layer_norm_fwd_task_impl();
TaskImplFunction get_layer_norm_bwd_task_impl();

OpTaskSignature get_layer_norm_init_signature();
OpTaskSignature get_layer_norm_fwd_signature();
OpTaskSignature get_layer_norm_bwd_signature();

OpTaskInvocation init(LayerNormAttrs const &);
OpTaskInvocation forward(LayerNormAttrs const &);
OpTaskInvocation backward(LayerNormAttrs const &);

} // namespace FlexFlow

#endif
