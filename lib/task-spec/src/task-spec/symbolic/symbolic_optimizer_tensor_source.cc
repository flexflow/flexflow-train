#include "task-spec/symbolic/symbolic_optimizer_tensor_source.h"

namespace FlexFlow {

int SymbolicOptimizerTensorSource::next_available_symbolic_optimizer_tensor_id = 0;

SymbolicOptimizerTensorSource::SymbolicOptimizerTensorSource() {}

symbolic_optimizer_tensor_guid_t SymbolicOptimizerTensorSource::new_symbolic_optimizer_tensor() {
  return symbolic_optimizer_tensor_guid_t{
      SymbolicOptimizerTensorSource::next_available_symbolic_optimizer_tensor_id++};
}

void SymbolicOptimizerTensorSource::reset() {
  SymbolicOptimizerTensorSource::next_available_symbolic_optimizer_tensor_id = 0;
}

} // namespace FlexFlow
