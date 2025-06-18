#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_SIGNATURE_IMPL_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_SIGNATURE_IMPL_H

#include "op-attrs/computation_graph_op_attrs.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/task_id_t.dtg.h"
#include "task-spec/task_signature_impl.dtg.h"

namespace FlexFlow {

TaskSignatureAndImpl get_task_sig_impl(task_id_t const &);
std::vector<task_id_t> get_task_ids(ComputationGraphOpAttrs const &);

OpTaskInvocation init(ComputationGraphOpAttrs const &);
OpTaskInvocation forward(ComputationGraphOpAttrs const &);
OpTaskInvocation backward(ComputationGraphOpAttrs const &);

} // namespace FlexFlow

#endif
