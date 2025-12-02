#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_INVOCATION_RUNTIME_ARG_REF_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_RUNTIME_TASK_INVOCATION_RUNTIME_ARG_REF_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/profiling_settings.dtg.h"
#include "pcg/device_type.dtg.h"
#include "task-spec/arg_ref.h"
#include "task-spec/device_specific.h"
#include "task-spec/ff_config.dtg.h"
#include "task-spec/ff_iteration_config.dtg.h"
#include "task-spec/per_device_op_state.dtg.h"
#include "task-spec/runtime_task_invocation/runtime_arg_ref_type.dtg.h"

namespace FlexFlow {

template <typename T>
using RuntimeArgRef = ArgRef<RuntimeArgRefType, T>;

RuntimeArgRef<ProfilingSettings> profiling_settings();
RuntimeArgRef<DeviceSpecific<device_handle_t>> ff_handle();
RuntimeArgRef<FFIterationConfig> iteration_config();
RuntimeArgRef<DeviceType> kernel_device_type();
RuntimeArgRef<PerDeviceOpState> per_device_op_state_for_layer(symbolic_layer_guid_t);

} // namespace FlexFlow

#endif
