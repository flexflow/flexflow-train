#include "task-spec/symbolic_loss_tensor_source.h"

namespace FlexFlow {

nonnegative_int SymbolicLossTensorSource::next_available_symbolic_loss_tensor_id = 0_n;

SymbolicLossTensorSource::SymbolicLossTensorSource() {}

symbolic_loss_tensor_guid_t SymbolicLossTensorSource::new_symbolic_loss_tensor() {
  return symbolic_loss_tensor_guid_t{SymbolicLossTensorSource::next_available_symbolic_loss_tensor_id++};
}

} // namespace FlexFlow
