#include "task-spec/runtime_arg_config.h"
#include "kernels/device_handle_t.h"

namespace FlexFlow {

RuntimeArgConfig
    cpu_make_runtime_arg_config(EnableProfiling enable_profiling,
                                ProfilingSettings profiling_settings) {
  return RuntimeArgConfig{
      DeviceSpecific<device_handle_t>::create(cpu_make_device_handle_t()),
      enable_profiling,
      profiling_settings,
      DeviceType::CPU,
  };
}

RuntimeArgConfig
    gpu_make_runtime_arg_config(PerDeviceFFHandle const &ff_handle,
                                EnableProfiling enable_profiling,
                                ProfilingSettings profiling_settings) {
  return RuntimeArgConfig{
      DeviceSpecific<device_handle_t>::create(
          gpu_make_device_handle_t(ff_handle)),
      enable_profiling,
      profiling_settings,
      DeviceType::GPU,
  };
}

} // namespace FlexFlow
