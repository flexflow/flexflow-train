#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_ARG_CONFIG_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_ARG_CONFIG_H

#include "task-spec/runtime_arg_config.dtg.h"

namespace FlexFlow {

RuntimeArgConfig
    cpu_make_runtime_arg_config(EnableProfiling enable_profiling,
                                ProfilingSettings profiling_settings);
RuntimeArgConfig
    gpu_make_runtime_arg_config(PerDeviceFFHandle const &ff_handle,
                                EnableProfiling enable_profiling,
                                ProfilingSettings profiling_settings);

} // namespace FlexFlow

#endif
