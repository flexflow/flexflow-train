#include "task-spec/symbolic_forward_tensor_source.h"

namespace FlexFlow {

int SymbolicForwardTensorSource::next_available_symbolic_forward_tensor_id = 0;

SymbolicForwardTensorSource::SymbolicForwardTensorSource() {}

symbolic_forward_tensor_guid_t SymbolicForwardTensorSource::new_symbolic_forward_tensor() {
  return symbolic_forward_tensor_guid_t{
      SymbolicForwardTensorSource::next_available_symbolic_forward_tensor_id++};
}

void SymbolicForwardTensorSource::reset() {
  SymbolicForwardTensorSource::next_available_symbolic_forward_tensor_id = 0;
}

} // namespace FlexFlow
