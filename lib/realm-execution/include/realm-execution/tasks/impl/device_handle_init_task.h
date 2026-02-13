#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_DEVICE_HANDLE_INIT_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_DEVICE_HANDLE_INIT_TASK_H

#include "realm-execution/device_specific_managed_per_device_ff_handle.h"
#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"

namespace FlexFlow {

void device_handle_init_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

Realm::Event spawn_device_handle_init_task(
    RealmContext &ctx,
    Realm::Processor target_proc,
    size_t workSpaceSize,
    bool allowTensorOpMathConversion,
    DeviceSpecificManagedPerDeviceFFHandle *result_ptr,
    Realm::Event precondition);

} // namespace FlexFlow

#endif
