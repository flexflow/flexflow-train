#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_DEVICE_STATE_INIT_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_DEVICE_STATE_INIT_TASK_H

#include "realm-execution/tasks/impl/device_state_init_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_device_state_init_task_args.dtg.h"

namespace FlexFlow {

SerializableDeviceStateInitTaskArgs device_state_init_task_args_to_serializable(
    DeviceStateInitTaskArgs const &);
DeviceStateInitTaskArgs device_state_init_task_args_from_serializable(
    SerializableDeviceStateInitTaskArgs const &);

} // namespace FlexFlow

#endif
