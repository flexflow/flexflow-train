#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_CONTROLLER_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_CONTROLLER_TASK_H

#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"

namespace FlexFlow {

void controller_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

Realm::Event
    collective_spawn_controller_task(RealmContext &ctx,
                                     Realm::Processor &target_proc,
                                     std::function<void(RealmContext &)> thunk);

} // namespace FlexFlow

#endif
