#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_TASK_REGISTRY_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_TASK_REGISTRY_H

#include "realm-execution/realm.h"
#include "realm-execution/tasks/task_id_t.dtg.h"

namespace FlexFlow {

/**
 * \brief Registers a function as a Realm task.
 *
 * \warning The event returned by this function <em>must be consumed</em> or
 * else Realm may not shut down properly.
 */
[[nodiscard]] Realm::Event register_task(Realm::Processor::Kind target_kind,
                                         task_id_t func_id,
                                         void (*task_body)(void const *,
                                                           size_t,
                                                           void const *,
                                                           size_t,
                                                           Realm::Processor));

/**
 * \brief Registers all known tasks (using \ref register_task).
 *
 * \warning The event returned by this function <em>must be consumed</em> or
 * else Realm may not shut down properly.
 */
[[nodiscard]] Realm::Event register_all_tasks();

} // namespace FlexFlow

#endif
