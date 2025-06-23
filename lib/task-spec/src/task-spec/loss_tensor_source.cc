#include "task-spec/loss_tensor_source.h"

namespace FlexFlow {

nonnegative_int LossTensorSource::next_available_loss_tensor_id = 0_n;

LossTensorSource::LossTensorSource() {}

loss_tensor_guid_t LossTensorSource::new_loss_tensor() {
  return loss_tensor_guid_t{LossTensorSource::next_available_loss_tensor_id++};
}

} // namespace FlexFlow
