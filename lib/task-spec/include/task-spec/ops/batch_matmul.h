#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_BATCH_MATMUL_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_BATCH_MATMUL_H

#include "task-spec/task_impl_function.dtg.h"
#include "op-attrs/ops/batch_matmul_attrs.dtg.h"
#include "task-spec/op_task_invocation.h"
#include "task-spec/op_task_signature.h"

namespace FlexFlow {

std::vector<task_id_t> get_task_ids(BatchMatmulAttrs const &);

TaskImplFunction get_batch_matmul_fwd_task_impl();
TaskImplFunction get_batch_matmul_bwd_task_impl();

OpTaskSignature get_batch_matmul_fwd_signature();
OpTaskSignature get_batch_matmul_bwd_signature();

OpTaskInvocation forward(BatchMatmulAttrs const &);
OpTaskInvocation backward(BatchMatmulAttrs const &);

} // namespace FlexFlow

#endif
