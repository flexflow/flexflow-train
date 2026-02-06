#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_TASK_REGISTRY_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_TASK_REGISTRY_H

#include "realm-execution/realm.h"
#include "realm-execution/task_id_t.dtg.h"

namespace FlexFlow {

[[nodiscard]] Realm::Event register_task(Realm::Processor::Kind target_kind,
                                         task_id_t func_id,
                                         void (*task_body)(void const *,
                                                           size_t,
                                                           void const *,
                                                           size_t,
                                                           Realm::Processor));

[[nodiscard]] Realm::Event register_all_tasks();

} // namespace FlexFlow

#endif
