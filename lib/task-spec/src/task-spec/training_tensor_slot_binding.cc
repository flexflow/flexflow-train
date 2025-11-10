#include "task-spec/training_tensor_slot_binding.h"
#include "utils/exception.h"

namespace FlexFlow {

TrainingTensorSlotBinding
  lower_fwb_tensor_binding_to_training_tensor_binding(
    SymbolicLayerTrainingTensorGroupSignature const &signature,
    FwbTensorSlotBinding const &fwb_slot_binding) {

  // TODO(@lockshaw)(#pr): 
  NOT_IMPLEMENTED();
  // fwb_tensor_slot_id_t fwb_slot_id = fwb_slot_binding.slot;
  // OpTensorSpec op_tensor_spec = fwb_slot_binding.bound; 
  //
  // SymbolicTrainingTensorGroup group = get_training_tensor_group_for_role_and_index(
  //     signature, op_tensor_spec.role, op_tensor_spec.idx);
  //
  // training_tensor_slot_id_t training_tensor_slot = 
  //   training_tensor_slot_from_fwb_slot(fwb_slot_id);
  //
  // symbolic_training_tensor_guid_t training_tensor = [&]() -> symbolic_training_tensor_guid_t {
  //   switch (fwb_slot_id.is_grad) {
  //     case IsGrad::NO:
  //       return symbolic_training_tensor_guid_t{
  //         group.forward_tensor,
  //       };
  //     case IsGrad::YES:
  //       return symbolic_training_tensor_guid_t{
  //         group.gradient_tensor,
  //       };
  //     default:
  //       PANIC("Invalid value for IsGrad {}", fwb_slot_id.is_grad);
  //   } 
  // }();
  //
  // return TrainingTensorSlotBinding{
  //   training_tensor_slot, 
  //   training_tensor,
  // };
}

} // namespace FlexFlow
