#include "op-attrs/tensor_slot_name.h"

namespace FlexFlow {

std::vector<TensorSlotName> get_variadic_inputs_slot_name_sequence() {
  return std::vector{
    TensorSlotName::INPUT_0,
    TensorSlotName::INPUT_1,
    TensorSlotName::INPUT_2,
    TensorSlotName::INPUT_3,
    TensorSlotName::INPUT_4,
    TensorSlotName::INPUT_5,
    TensorSlotName::INPUT_6,
    TensorSlotName::INPUT_7,
  };
};

std::vector<TensorSlotName> get_variadic_outputs_slot_name_sequence() {
  return std::vector{
    TensorSlotName::OUTPUT_0,
    TensorSlotName::OUTPUT_1,
    TensorSlotName::OUTPUT_2,
    TensorSlotName::OUTPUT_3,
    TensorSlotName::OUTPUT_4,
    TensorSlotName::OUTPUT_5,
  };
}

} // namespace FlexFlow
