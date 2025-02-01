#include "local-execution/lowered_tensor_source.h"

namespace FlexFlow {

size_t LoweredTensorSource::next_available_lowered_tensor_id = 0;

LoweredTensorSource::LoweredTensorSource() {}

lowered_tensor_t LoweredTensorSource::new_lowered_tensor() {
  return lowered_tensor_t{
      LoweredTensorSource::next_available_lowered_tensor_id++};
}

} // namespace FlexFlow
