#ifndef _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_SLOT_NAME_H
#define _FLEXFLOW_LIB_OP_ATTRS_INCLUDE_OP_ATTRS_TENSOR_SLOT_NAME_H

#include "op-attrs/tensor_slot_name.dtg.h"

namespace FlexFlow {

std::vector<TensorSlotName> get_variadic_inputs_slot_name_sequence();
std::vector<TensorSlotName> get_variadic_outputs_slot_name_sequence();

} // namespace FlexFlow

#endif
