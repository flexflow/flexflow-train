#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_DEVICE_HANDLE_INIT_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_DEVICE_HANDLE_INIT_TASK_H

#include "realm-execution/tasks/impl/device_handle_init_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_device_handle_init_task_args.dtg.h"

namespace FlexFlow {

SerializableDeviceHandleInitTaskArgs
    device_handle_init_task_args_to_serializable(
        DeviceHandleInitTaskArgs const &);
DeviceHandleInitTaskArgs device_handle_init_task_args_from_serializable(
    SerializableDeviceHandleInitTaskArgs const &);

} // namespace FlexFlow

#endif
