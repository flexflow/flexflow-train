#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_PER_DEVICE_OP_STATE_INIT_RETURN_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_PER_DEVICE_OP_STATE_INIT_RETURN_TASK_H

#include "realm-execution/device_specific_ptr.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"
#include "task-spec/per_device_op_state.dtg.h"

namespace FlexFlow {

/**
 * \brief The function registered as a %Realm task for returning the
 * asynchronously-initialized PerDeviceOpState. Dispatched by \ref
 * spawn_per_device_op_state_init_return_task.
 *
 * To understand how this fits into the broader structure of \ref
 * realm-execution, see \ref realm-execution-tasks.
 */
void per_device_op_state_init_return_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Launches the task (\ref per_device_op_state_init_return_task_body) for returning
 * the asynchronously-initialized PerDeviceOpState.
 *
 * To understand how this fits into the broader structure of \ref
 * realm-execution, see \ref realm-execution-tasks.
 */
Realm::Event spawn_per_device_op_state_init_return_task(
    RealmContext &ctx,
    Realm::Processor origin_proc,
    DeviceSpecificPtr<PerDeviceOpState> const &result,
    DeviceSpecificPtr<PerDeviceOpState> *origin_result_ptr,
    Realm::Event precondition);

} // namespace FlexFlow

#endif
