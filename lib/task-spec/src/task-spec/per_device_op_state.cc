#include "task-spec/per_device_op_state.h"
#include "utils/overload.h"

namespace FlexFlow {

PerDeviceOpState get_device_state_from_device_specific(
    DeviceSpecificPerDeviceOpState const &device_specific,
    device_id_t device_idx) {
  return device_specific.visit<PerDeviceOpState>(
      [&](auto const &x) { return PerDeviceOpState{*(x.get(device_idx))}; });
}

} // namespace FlexFlow
