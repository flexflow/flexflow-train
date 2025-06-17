#include "task-spec/task_invocation.h"
#include "task-spec/task_arg_spec.h"
#include "utils/containers/keys.h"

namespace FlexFlow {

bool is_invocation_valid(TaskSignature const &sig, TaskInvocation const &inv) {
  TaskBinding binding = inv.binding;

  for (std::pair<slot_id_t, TaskArgSpec> const &arg_binding :
       binding.get_arg_bindings()) {
    if (sig.task_arg_types.count(arg_binding.first)) {
      if (get_type_index(arg_binding.second) !=
          sig.task_arg_types.at(arg_binding.first)) {
        return false; // incorrect arg type
      }
    } else {
      return false; // slot doesn't exist in signature
    }
  }

  for (std::pair<tensor_sub_slot_id_t, training_tensor_guid_t> const &tensor_binding :
       binding.get_tensor_bindings()) {
    slot_id_t tensor_slot_id = tensor_binding.first.slot_id;
    if (sig.tensor_guid_slots.count(tensor_slot_id)) {
      if (tensor_binding.first.tensor_type ==
          sig.tensor_guid_slots.at(tensor_slot_id).tensor_type) {
        return false; // incorrect tensor type
      }
    } else {
      return false; // slot doesn't exist in signature
    }
  }

  return true;
}

} // namespace FlexFlow
