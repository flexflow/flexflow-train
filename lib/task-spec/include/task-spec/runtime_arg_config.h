#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_ARG_CONFIG_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_ARG_CONFIG_H

#include "task-spec/concrete_arg_spec.h"
#include "task-spec/runtime_arg_config.dtg.h"
#include "task-spec/slot_id_t.dtg.h"
#include "task-spec/task_binding.h"

namespace FlexFlow {

RuntimeArgConfig
    cpu_make_runtime_arg_config(EnableProfiling enable_profiling,
                                ProfilingSettings profiling_settings);
RuntimeArgConfig
    gpu_make_runtime_arg_config(PerDeviceFFHandle const &ff_handle,
                                EnableProfiling enable_profiling,
                                ProfilingSettings profiling_settings);

std::unordered_map<slot_id_t, ConcreteArgSpec>
    construct_arg_slots_backing(TaskBinding const &, RuntimeArgConfig const &);


} // namespace FlexFlow

#endif
