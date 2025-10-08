#include "task-spec/runtime_arg_config.h"
#include "kernels/device_handle_t.h"
#include "task-spec/lower_op_task_invocation_to_runtime_task_invocation.h"
#include "utils/containers/map_values.h"
#include "utils/overload.h"

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

std::unordered_map<slot_id_t, ConcreteArgSpec>
    construct_arg_slots_backing(RuntimeTaskBinding const &binding,
                                RuntimeArgConfig const &runtime_arg_config) {
  return map_values(
      binding.get_arg_bindings(), 
      [&](RuntimeArgSpec const &arg_binding) -> ConcreteArgSpec {
        return arg_binding.template visit<ConcreteArgSpec>(
            overload{
              [&](RuntimeArgRefSpec const &s) {
                 return lower_runtime_arg_ref_spec_to_concrete_arg_spec(s, runtime_arg_config);
               },
               [](ConcreteArgSpec const &s) { return s; },
             });
      });
  ;
}


} // namespace FlexFlow
