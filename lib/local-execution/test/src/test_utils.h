#ifndef _FLEXFLOW_LOCAL_EXECUTION_TEST_UTILS
#define _FLEXFLOW_LOCAL_EXECUTION_TEST_UTILS

#include "kernels/ff_handle.h"
#include "pcg/tensor_guid_t.dtg.h"

namespace FlexFlow {

struct MockTensorGuidSource {
public:
  MockTensorGuidSource();

  tensor_guid_t new_mock_tensor_guid();

private:
  static size_t next_available_mock_tensor_guid;
};

PerDeviceFFHandle get_mock_per_device_ff_handle();

} // namespace FlexFlow

#endif
