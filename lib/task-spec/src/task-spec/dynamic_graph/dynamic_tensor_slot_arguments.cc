#include "task-spec/dynamic_graph/dynamic_tensor_slot_arguments.h"
#include "utils/containers/are_all_same.h"

namespace FlexFlow {

bool slot_arguments_satisfies_expansion_conditions(
  DynamicTensorSlotArguments const &slot_arguments,
  std::function<bool(DynamicTensorSlotArguments const &)> const &args_condition,
  std::function<bool(DynamicValueAttrs const &)> const &value_condition) {

  bool from_slot_args = args_condition(slot_arguments);

  if (slot_arguments.values.has_value()) {
    std::vector<bool> values_are_pass_expanded  
      = transform(slot_arguments.values.value(), value_condition);

    ASSERT(are_all_same(values_are_pass_expanded));
    bool from_values = values_are_pass_expanded.at(0);

    ASSERT(from_values == from_slot_args);
  }
  return from_slot_args;
}

} // namespace FlexFlow
