#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_CONTROLLER_TASK_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_IMPL_CONTROLLER_TASK_H

#include "realm-execution/realm.h"
#include "realm-execution/realm_context.h"

namespace FlexFlow {

/**
 * \brief A stub function to work around Realm not allowing lambdas to be be registered as Realm tasks.
 * Takes the desired lambda to run as the \ref term-controller as an argument and immediately calls it.
 */
void controller_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

/**
 * \brief Dispatches the \ref controller task. Packages up the provided \ref std::function and
 * passes it along to \ref controller_task_body.
 */
Realm::Event
    collective_spawn_controller_task(RealmContext &ctx,
                                     Realm::Processor &target_proc,
                                     std::function<void(RealmContext &)> thunk,
                                     Realm::Event precondition);

} // namespace FlexFlow

#endif
