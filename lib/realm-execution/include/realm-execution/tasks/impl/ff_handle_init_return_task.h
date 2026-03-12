#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_DEVICE_HANDLE_INIT_RETURN_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_DEVICE_HANDLE_INIT_RETURN_TASK_H

#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"

namespace FlexFlow {

/**
 * \brief The function registered as a Realm task for returning the
 * asynchronously-initialized \ref PerDeviceFFHandle. Dispatched by \ref
 * spawn_ff_handle_init_return_task.
 *
 * To understand how this fits into the broader structure of \ref
 * realm-execution, see \ref realm-execution-tasks.
 */
void ff_handle_init_return_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Launches the task (\ref ff_handle_init_return_task_body) for returning
 * the asynchronously-initialized \ref PerDeviceFFHandle.
 *
 * \param ctx The Realm context.
 * \param origin_proc The processor to send the result to.
 * \param result The result value.
 * \param origin_result_ptr The pointer, on the origin processor, to which the
 * result should be written.
 * \param precondition Event precondition to the resulting task.
 *
 * To understand how this fits into the broader structure of \ref
 * realm-execution, see \ref realm-execution-tasks.
 */
Realm::Event spawn_ff_handle_init_return_task(
    RealmContext &ctx,
    Realm::Processor origin_proc,
    DeviceSpecificPtr<ManagedPerDeviceFFHandle> const &result,
    DeviceSpecificPtr<ManagedPerDeviceFFHandle> *origin_result_ptr,
    Realm::Event precondition);

} // namespace FlexFlow

#endif
