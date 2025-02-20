#ifndef _FLEXFLOW_LOCAL_EXECUTION_TASK_SIGNATURE_IMPL_H
#define _FLEXFLOW_LOCAL_EXECUTION_TASK_SIGNATURE_IMPL_H

#include "local-execution/task_signature_impl.dtg.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_id_t.dtg.h"

namespace FlexFlow {

TaskSignatureAndImpl get_task_sig_impl(task_id_t const &);
std::vector<task_id_t> get_task_ids(ComputationGraphOpAttrs const &);

OpTaskInvocation init(ComputationGraphOpAttrs const &);
OpTaskInvocation forward(ComputationGraphOpAttrs const &);
OpTaskInvocation backward(ComputationGraphOpAttrs const &);

} // namespace FlexFlow

#endif
