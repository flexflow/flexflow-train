#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_OP_TENSOR_SPEC_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_OPS_OP_TENSOR_SPEC_H

#include "task-spec/ops/op_tensor_spec.dtg.h"

namespace FlexFlow {

OpTensorSpec input_tensor(nonnegative_int idx,
                          OpSlotOptions option = OpSlotOptions::NECESSARY);
OpTensorSpec output_tensor(nonnegative_int idx,
                           OpSlotOptions option = OpSlotOptions::NECESSARY);
OpTensorSpec weight_tensor(nonnegative_int idx,
                           OpSlotOptions option = OpSlotOptions::NECESSARY);

} // namespace FlexFlow

#endif
