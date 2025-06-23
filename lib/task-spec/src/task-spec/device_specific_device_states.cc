#include "task-spec/device_specific_device_states.h"
#include "kernels/mha_per_device_state.dtg.h"

namespace FlexFlow {

using T = MHAPerDeviceState;

template std::optional<DeviceSpecificDeviceStates>
    make_device_specific_state(std::optional<T> const &);

} // namespace FlexFlow
