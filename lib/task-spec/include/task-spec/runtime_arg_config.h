#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_ARG_CONFIG_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_ARG_CONFIG_H

#include "task-spec/concrete_arg_spec.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/slot_id_t.dtg.h"
#include "task-spec/runtime_task_binding.h"

namespace FlexFlow {

RuntimeArgConfig
    cpu_make_runtime_arg_config(device_id_t device_id,
                                EnableProfiling enable_profiling,
                                ProfilingSettings profiling_settings);
RuntimeArgConfig
    gpu_make_runtime_arg_config(device_id_t device_id,
                                PerDeviceFFHandle const &ff_handle,
                                EnableProfiling enable_profiling,
                                ProfilingSettings profiling_settings);

} // namespace FlexFlow

#endif
