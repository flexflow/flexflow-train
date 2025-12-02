#include "task-spec/dynamic_graph/dynamic_tensor_slot.h"

namespace FlexFlow {

DynamicTensorSlot decide_tensor_slot_role(DynamicTensorSlot const &slot, DynamicTensorRole role) {
  ASSERT(slot.slot_tensor_role == std::nullopt);

  DynamicTensorSlot result = slot;
  result.slot_tensor_role = role;

  return result;
}

} // namespace FlexFlow
