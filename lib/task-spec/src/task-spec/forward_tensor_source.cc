#include "task-spec/forward_tensor_source.h"

namespace FlexFlow {

int ForwardTensorSource::next_available_forward_tensor_id = 0;

ForwardTensorSource::ForwardTensorSource() {}

forward_tensor_guid_t ForwardTensorSource::new_forward_tensor() {
  return forward_tensor_guid_t{
      ForwardTensorSource::next_available_forward_tensor_id++};
}

void ForwardTensorSource::reset() {
  ForwardTensorSource::next_available_forward_tensor_id = 0;
}

} // namespace FlexFlow
