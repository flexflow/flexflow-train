#include "task-spec/optimizer_tensor_source.h"

namespace FlexFlow {

int OptimizerTensorSource::next_available_optimizer_tensor_id = 0;

OptimizerTensorSource::OptimizerTensorSource() {}

optimizer_tensor_guid_t OptimizerTensorSource::new_optimizer_tensor() {
  return optimizer_tensor_guid_t{
      OptimizerTensorSource::next_available_optimizer_tensor_id++};
}

void OptimizerTensorSource::reset() {
  OptimizerTensorSource::next_available_optimizer_tensor_id = 0;
}

} // namespace FlexFlow
