#include "task-spec/runtime_task_signature.h"

namespace FlexFlow {

RuntimeTaskSignature make_empty_runtime_task_signature() {
  return RuntimeTaskSignature(std::nullopt, {}, {});
}

void add_slot(RuntimeTaskSignature &task_signature,
              int name,
              TrainingTensorType tensor_type,
              SlotType slot_type) {
  add_slot(task_signature, slot_id_t{name}, tensor_type, slot_type);
}

void add_slot(RuntimeTaskSignature &task_signature,
              slot_id_t name,
              TrainingTensorType tensor_type,
              SlotType slot_type) {
  TensorTypeSlotSpec tensor_guid_slot_spec =
      TensorTypeSlotSpec{name, tensor_type, slot_type};
  task_signature.tensor_guid_slots.insert({name, tensor_guid_slot_spec});
}

} // namespace FlexFlow
