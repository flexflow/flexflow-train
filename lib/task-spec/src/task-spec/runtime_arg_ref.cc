#include "task-spec/runtime_arg_ref.h"
#include "kernels/device_handle_t.dtg.h"
#include "task-spec/device_specific.h"

namespace FlexFlow {

RuntimeArgRef<ProfilingSettings> profiling_settings() {
  return {RuntimeArgRefType::PROFILING_SETTINGS};
}

RuntimeArgRef<DeviceSpecific<device_handle_t>> ff_handle() {
  return {RuntimeArgRefType::FF_HANDLE};
}

RuntimeArgRef<FFIterationConfig> iteration_config() {
  return {RuntimeArgRefType::FF_ITERATION_CONFIG};
}

RuntimeArgRef<DeviceType> kernel_device_type() {
  return {RuntimeArgRefType::KERNEL_DEVICE_TYPE};
}

} // namespace FlexFlow
