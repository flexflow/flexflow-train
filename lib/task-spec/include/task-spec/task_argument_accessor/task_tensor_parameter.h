#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_ARGUMENT_ACCESSOR_TASK_TENSOR_PARAMETER_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_TASK_ARGUMENT_ACCESSOR_TASK_TENSOR_PARAMETER_H

#include "task-spec/task_argument_accessor/task_tensor_parameter.dtg.h"

namespace FlexFlow {

TaskTensorParameter make_task_tensor_parameter_fwd(TensorSlotName);
TaskTensorParameter make_task_tensor_parameter_grad(TensorSlotName);
TaskTensorParameter make_task_tensor_parameter_opt(TensorSlotName,
                                                   OptimizerSlotName);
TaskTensorParameter make_task_tensor_parameter_loss();

} // namespace FlexFlow

#endif
