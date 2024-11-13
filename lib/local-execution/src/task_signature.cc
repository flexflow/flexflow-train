#include "local-execution/task_signature.h"

namespace FlexFlow {

TaskSignature make_empty_task_signature() {
  return TaskSignature(std::nullopt, {}, {});
}

void add_slot(TaskSignature &task_signature,
              int name,
              TensorType tensor_type,
              SlotType slot_type) {
  add_slot(task_signature, slot_id_t{name}, tensor_type, slot_type);
}

void add_slot(TaskSignature &task_signature,
              slot_id_t name,
              TensorType tensor_type,
              SlotType slot_type) {
  TensorTypeSlotSpec tensor_guid_slot_spec =
      TensorTypeSlotSpec{slot_type, tensor_type};
  task_signature.tensor_guid_slots.insert({name, tensor_guid_slot_spec});
}

} // namespace FlexFlow
