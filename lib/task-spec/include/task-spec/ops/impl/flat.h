#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_FLAT_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_FLAT_H

#include "op-attrs/ops/flat_attrs.dtg.h"
#include "task-spec/ops/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_flat_fwd_task_impl();
TaskImplFunction get_flat_bwd_task_impl();

OpTaskSignature get_flat_fwd_signature();
OpTaskSignature get_flat_bwd_signature();

OpTaskInvocation forward(FlatAttrs const &);
OpTaskInvocation backward(FlatAttrs const &);

} // namespace FlexFlow

#endif
