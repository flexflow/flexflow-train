#include "task-spec/symbolic/symbolic_gradient_tensor_source.h"

namespace FlexFlow {

int SymbolicGradientTensorSource::next_available_symbolic_gradient_tensor_id = 0;

SymbolicGradientTensorSource::SymbolicGradientTensorSource() {}

symbolic_gradient_tensor_guid_t SymbolicGradientTensorSource::new_symbolic_gradient_tensor() {
  return symbolic_gradient_tensor_guid_t{
      SymbolicGradientTensorSource::next_available_symbolic_gradient_tensor_id++};
}

void SymbolicGradientTensorSource::reset() {
  SymbolicGradientTensorSource::next_available_symbolic_gradient_tensor_id = 0;
}

} // namespace FlexFlow
