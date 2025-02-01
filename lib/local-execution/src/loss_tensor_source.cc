#include "local-execution/loss_tensor_source.h"

namespace FlexFlow {

size_t LossTensorSource::next_available_loss_tensor_id = 0;

LossTensorSource::LossTensorSource() {}

loss_tensor_t LossTensorSource::new_loss_tensor() {
  return loss_tensor_t{LossTensorSource::next_available_loss_tensor_id++};
}

} // namespace FlexFlow
