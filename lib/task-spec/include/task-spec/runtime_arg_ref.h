#ifndef _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_RUNTIME_ARG_REF_H
#define _FLEXFLOW_RUNTIME_SRC_TASK_SPEC_RUNTIME_ARG_REF_H

#include "pcg/device_type.dtg.h"
#include "task-spec/arg_ref.h"
#include "task-spec/config.h"
#include "task-spec/device_specific.h"
#include "kernels/profiling_settings.dtg.h"
#include "task-spec/runtime_arg_ref_type.dtg.h"

namespace FlexFlow {

template <typename T>
using RuntimeArgRef = ArgRef<RuntimeArgRefType, T>;

using RuntimeArgRefSpec = ArgRefSpec<RuntimeArgRefType>;

RuntimeArgRef<ProfilingSettings> profiling_settings();
RuntimeArgRef<DeviceSpecific<PerDeviceFFHandle>> ff_handle();
RuntimeArgRef<FFIterationConfig> iteration_config();
RuntimeArgRef<DeviceType> kernel_device_type();

} // namespace FlexFlow

#endif
