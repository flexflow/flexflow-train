#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_TENSOR_SLOT_ARGUMENTS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DYNAMIC_GRAPH_DYNAMIC_TENSOR_SLOT_ARGUMENTS_H

#include "task-spec/dynamic_graph/dynamic_tensor_slot_arguments.dtg.h"

namespace FlexFlow {

bool slot_arguments_satisfies_expansion_conditions(
  DynamicTensorSlotArguments const &slot_arguments,
  std::function<bool(DynamicTensorSlotArguments const &)> const &args_condition,
  std::function<bool(DynamicValueAttrs const &)> const &value_condition);

} // namespace FlexFlow

#endif
