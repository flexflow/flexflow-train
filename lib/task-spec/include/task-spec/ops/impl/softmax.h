#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_SOFTMAX_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_IMPL_SOFTMAX_H

#include "op-attrs/ops/softmax_attrs.dtg.h"
#include "task-spec/ops/op_task_invocation.h"
#include "task-spec/task_impl_function.dtg.h"

namespace FlexFlow {

TaskImplFunction get_softmax_init_task_impl();
TaskImplFunction get_softmax_fwd_task_impl();
TaskImplFunction get_softmax_bwd_task_impl();

OpTaskInvocation init(SoftmaxAttrs const &);
OpTaskInvocation forward(SoftmaxAttrs const &);
OpTaskInvocation backward(SoftmaxAttrs const &);

} // namespace FlexFlow

#endif
