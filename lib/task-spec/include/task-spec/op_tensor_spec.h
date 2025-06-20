#ifndef _FLEXFLOW_LOCAL_EXECUTION_OP_TENSOR_SPEC_REF_H
#define _FLEXFLOW_LOCAL_EXECUTION_OP_TENSOR_SPEC_REF_H

#include "task-spec/op_tensor_spec.dtg.h"

namespace FlexFlow {

OpTensorSpec input_tensor(nonnegative_int idx, OpSlotOptions option = OpSlotOptions::NECESSARY);
OpTensorSpec output_tensor(nonnegative_int idx,
                           OpSlotOptions option = OpSlotOptions::NECESSARY);
OpTensorSpec weight_tensor(nonnegative_int idx,
                           OpSlotOptions option = OpSlotOptions::NECESSARY);

} // namespace FlexFlow

#endif
