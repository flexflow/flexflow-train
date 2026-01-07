#include "task-spec/dynamic_graph/dynamic_tensor_role.h"

namespace FlexFlow {

DynamicTensorRole
    dynamic_tensor_role_from_fwb_tensor_type(FwbTensorType tensor_type) {
  return DynamicTensorRole{tensor_type};
}

DynamicTensorRole mk_dynamic_tensor_role_fwd() {
  return DynamicTensorRole{FwbTensorType::FORWARD};
}

DynamicTensorRole mk_dynamic_tensor_role_bwd() {
  return DynamicTensorRole{FwbTensorType::GRADIENT};
}

DynamicTensorRole mk_dynamic_tensor_role_opt(OptimizerSlotName s) {
  return DynamicTensorRole{DynamicOptimizerTensorRole{s}};
}

} // namespace FlexFlow
