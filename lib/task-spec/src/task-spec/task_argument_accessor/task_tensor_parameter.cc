#include "task-spec/task_argument_accessor/task_tensor_parameter.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

TaskTensorParameter make_task_tensor_parameter_fwd(TensorSlotName slot) {
  return TaskTensorParameter{TaskForwardTensorParameter{slot}};
}

TaskTensorParameter make_task_tensor_parameter_grad(TensorSlotName slot) {
  return TaskTensorParameter{TaskGradientTensorParameter{slot}};
}

TaskTensorParameter make_task_tensor_parameter_opt(TensorSlotName slot,
                                                   OptimizerSlotName opt_slot) {
  return TaskTensorParameter{TaskOptimizerTensorParameter{slot, opt_slot}};
}

TaskTensorParameter make_task_tensor_parameter_loss() {
  return TaskTensorParameter{TaskLossTensorParameter{}};
}

} // namespace FlexFlow
