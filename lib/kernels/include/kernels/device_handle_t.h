#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DEVICE_HANDLE_T_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_DEVICE_HANDLE_T_H

#include "kernels/device_handle_t.dtg.h"
#include "kernels/managed_per_device_ff_handle.h"

namespace FlexFlow {

device_handle_t device_handle_t_from_managed_handle(
    std::optional<ManagedPerDeviceFFHandle> const &managed_handle);

device_handle_t gpu_make_device_handle_t(PerDeviceFFHandle const &ff_handle);
device_handle_t cpu_make_device_handle_t();

} // namespace FlexFlow

#endif
