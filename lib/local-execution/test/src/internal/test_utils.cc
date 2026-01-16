#include "internal/test_utils.h"
#include "pcg/tensor_guid_t.dtg.h"

namespace FlexFlow {

PerDeviceFFHandle get_mock_per_device_ff_handle() {
  return {nullptr, nullptr, nullptr, 0, false};
}

size_t MockTensorGuidSource::next_available_mock_tensor_guid = 0;

MockTensorGuidSource::MockTensorGuidSource() {}

tensor_guid_t MockTensorGuidSource::new_mock_tensor_guid() {
  // FIXME (Elliott): where is the guid supposed to go now???
  size_t next_guid = MockTensorGuidSource::next_available_mock_tensor_guid++;
  return tensor_guid_t{KwargDataflowOutput{Node{0}, TensorSlotName::INPUT}};
}

} // namespace FlexFlow
