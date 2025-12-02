#include "task-spec/runtime_task_invocation/runtime_arg_ref.h"
#include "kernels/device_handle_t.dtg.h"
#include "task-spec/device_specific.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

RuntimeArgRef<ProfilingSettings> profiling_settings() {
  return {RuntimeArgRefType{
    ArgumentlessRuntimeArgRefType::PROFILING_SETTINGS
  }};
}

RuntimeArgRef<DeviceSpecific<device_handle_t>> ff_handle() {
  return {RuntimeArgRefType{
    ArgumentlessRuntimeArgRefType::FF_HANDLE
  }};
}

RuntimeArgRef<FFIterationConfig> iteration_config() {
  return {RuntimeArgRefType{
    ArgumentlessRuntimeArgRefType::FF_ITERATION_CONFIG,
  }};
}

RuntimeArgRef<DeviceType> kernel_device_type() {
  return {RuntimeArgRefType{
    ArgumentlessRuntimeArgRefType::KERNEL_DEVICE_TYPE
  }};
}

RuntimeArgRef<PerDeviceOpState> per_device_op_state_for_layer(symbolic_layer_guid_t layer) {
  return {RuntimeArgRefType{
    PerDeviceOpStateRuntimeArgRefType{
      layer,
    },
  }};
}

} // namespace FlexFlow
