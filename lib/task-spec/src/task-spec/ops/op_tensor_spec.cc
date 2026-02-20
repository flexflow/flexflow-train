#include "task-spec/ops/op_tensor_spec.h"

namespace FlexFlow {

OpTensorSpec input_tensor(nonnegative_int idx, OpSlotOptions option) {
  return OpTensorSpec{TensorRole::INPUT, option, idx};
}

OpTensorSpec output_tensor(nonnegative_int idx, OpSlotOptions option) {
  return OpTensorSpec{TensorRole::OUTPUT, option, idx};
}

OpTensorSpec weight_tensor(nonnegative_int idx, OpSlotOptions option) {
  return OpTensorSpec{TensorRole::WEIGHT, option, idx};
}

} // namespace FlexFlow
