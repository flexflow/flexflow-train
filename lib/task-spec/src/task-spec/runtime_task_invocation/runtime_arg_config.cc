#include "task-spec/runtime_task_invocation/runtime_arg_config.h"
#include "kernels/device_handle_t.h"
#include "task-spec/lower_op_task_invocation_to_runtime_task_invocation.h"
#include "utils/containers/map_values.h"
#include "utils/overload.h"

namespace FlexFlow {

RuntimeArgConfig
    cpu_make_runtime_arg_config(device_id_t device_id, 
                                EnableProfiling enable_profiling,
                                ProfilingSettings profiling_settings) {
  return RuntimeArgConfig{
      enable_profiling,
      profiling_settings,
      DeviceType::CPU,
  };
}

RuntimeArgConfig
    gpu_make_runtime_arg_config(device_id_t device_id, 
                                PerDeviceFFHandle const &ff_handle,
                                EnableProfiling enable_profiling,
                                ProfilingSettings profiling_settings) {
  return RuntimeArgConfig{
      enable_profiling,
      profiling_settings,
      DeviceType::GPU,
  };
}

} // namespace FlexFlow
