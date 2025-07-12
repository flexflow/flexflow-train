#include "task-spec/gradient_tensor_source.h"

namespace FlexFlow {

int GradientTensorSource::next_available_gradient_tensor_id = 0;

GradientTensorSource::GradientTensorSource() {}

gradient_tensor_guid_t GradientTensorSource::new_gradient_tensor() {
  return gradient_tensor_guid_t{
      GradientTensorSource::next_available_gradient_tensor_id++};
}

void GradientTensorSource::reset() {
  GradientTensorSource::next_available_gradient_tensor_id = 0;
}

} // namespace FlexFlow
