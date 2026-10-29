#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/get_standard_operator_task_group.h"
#include "op-attrs/standard_operator_task_group.h"

namespace FlexFlow {

OperatorTaskSpace get_operator_task_space(
    ComputationGraphOpAttrs const &attrs,
    std::map<TensorSlotName, ParallelTensorDimDegrees> const &inputs_degrees) {
  return task_space_for_standard_operator_task_group(
      get_standard_operator_task_group(attrs, inputs_degrees));
}

} // namespace FlexFlow
