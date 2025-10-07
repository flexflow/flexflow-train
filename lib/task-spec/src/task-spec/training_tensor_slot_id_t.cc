#include "task-spec/training_tensor_slot_id_t.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

training_tensor_slot_id_t
  training_tensor_slot_from_fwb_slot(fwb_tensor_slot_id_t fwb_slot_id) {

  if (fwb_slot_id.is_grad == IsGrad::NO) {
    return training_tensor_slot_id_t{
      fwb_slot_id.slot_id,
      TensorType::FORWARD,
    };
  } else if (fwb_slot_id.is_grad == IsGrad::YES) {
    return training_tensor_slot_id_t{
      fwb_slot_id.slot_id,
      TensorType::GRADIENT,
    };
  } else {
    PANIC("Invalid value for IsGrad {}", fwb_slot_id.is_grad);
  }
}

} // namespace FlexFlow
