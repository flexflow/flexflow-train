#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TENSOR_SPEC_REF_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TENSOR_SPEC_REF_H

#include "local-execution/op_task_signature.h"

namespace FlexFlow {

struct OpTensorSpec {
  TensorRole role;
  OpSlotOptions slot_option;
  req<int> idx;
};
FF_VISITABLE_STRUCT(OpTensorSpec, role, slot_option, idx);

OpTensorSpec input_tensor(int, OpSlotOptions option = OpSlotOptions::NECESSARY);
OpTensorSpec output_tensor(int,
                           OpSlotOptions option = OpSlotOptions::NECESSARY);
OpTensorSpec weight_tensor(int,
                           OpSlotOptions option = OpSlotOptions::NECESSARY);

} // namespace FlexFlow

#endif
