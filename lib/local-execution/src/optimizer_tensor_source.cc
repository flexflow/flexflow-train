#include "local-execution/optimizer_tensor_source.h"

namespace FlexFlow {

size_t OptimizerTensorSource::next_available_optimizer_tensor_id = 0;

OptimizerTensorSource::OptimizerTensorSource() {}

optimizer_tensor_t OptimizerTensorSource::new_optimizer_tensor() {
  return optimizer_tensor_t{
      OptimizerTensorSource::next_available_optimizer_tensor_id++};
}

} // namespace FlexFlow
