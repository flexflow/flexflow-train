#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_OP_TASK_INVOCATION_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_OP_TASK_INVOCATION_H

#include "op-attrs/computation_graph_op_attrs.dtg.h"
#include "task-spec/ops/op_task_invocation.dtg.h"
#include "task-spec/ops/op_task_type.dtg.h"

namespace FlexFlow {

std::optional<OpTaskInvocation> get_init_op_task_invocation(ComputationGraphOpAttrs const &);
std::optional<OpTaskInvocation>
    get_forward_op_task_invocation(ComputationGraphOpAttrs const &);
std::optional<OpTaskInvocation>
    get_backward_op_task_invocation(ComputationGraphOpAttrs const &);

std::optional<OpTaskInvocation> get_op_task_invocation(ComputationGraphOpAttrs const &, OpTaskType);

} // namespace FlexFlow

#endif
