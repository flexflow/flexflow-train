#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_DEVICE_HANDLE_INIT_TASK_ARGS_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_SERIALIZABLE_DEVICE_HANDLE_INIT_TASK_ARGS_H

#include "realm-execution/tasks/impl/ff_handle_init_task_args.dtg.h"
#include "realm-execution/tasks/impl/serializable_ff_handle_init_task_args.dtg.h"

namespace FlexFlow {

SerializableFfHandleInitTaskArgs
    ff_handle_init_task_args_to_serializable(
        FfHandleInitTaskArgs const &);

FfHandleInitTaskArgs ff_handle_init_task_args_from_serializable(
    SerializableFfHandleInitTaskArgs const &);

} // namespace FlexFlow

#endif
