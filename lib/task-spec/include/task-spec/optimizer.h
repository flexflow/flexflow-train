#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPTIMIZER_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPTIMIZER_H

#include "pcg/optimizer_attrs.dtg.h"
#include "pcg/optimizers/adam_optimizer_attrs.dtg.h"
#include "pcg/optimizers/sgd_optimizer_attrs.dtg.h"
#include "task-spec/task_impl_function.dtg.h"
#include "task-spec/task_invocation.dtg.h"
#include "task-spec/task_signature.h"

namespace FlexFlow {

TaskSignature get_update_signature(OptimizerAttrs const &);
TaskInvocation get_update_invocation(
    OptimizerAttrs const &,
    symbolic_forward_tensor_guid_t const &weight,
    symbolic_gradient_tensor_guid_t const &weight_grad,
    std::vector<symbolic_optimizer_tensor_guid_t> const &grad_buffer_tensors);
TaskImplFunction get_update_task_impl(OptimizerAttrs const &);

TaskSignature get_sgd_update_signature();
TaskInvocation sgd_update(SGDOptimizerAttrs const &,
                          symbolic_forward_tensor_guid_t const &weight,
                          symbolic_gradient_tensor_guid_t const &weight_grad,
                          symbolic_optimizer_tensor_guid_t const &sgd_v);
TaskImplFunction get_sgd_update_task_impl();

TaskSignature get_adam_update_signature();
TaskInvocation adam_update(AdamOptimizerAttrs const &,
                           symbolic_forward_tensor_guid_t const &weight,
                           symbolic_gradient_tensor_guid_t const &weight_grad,
                           symbolic_optimizer_tensor_guid_t const &adam_v,
                           symbolic_optimizer_tensor_guid_t const &adam_m);
TaskImplFunction get_adam_update_task_impl();

} // namespace FlexFlow

#endif
