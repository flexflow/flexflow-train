#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_TASKS_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_REALM_TASKS_H

#include "realm-execution/realm.h"

namespace FlexFlow {

void op_task_body(void const *, size_t, void const *, size_t, Realm::Processor);

void controller_task_body(
    void const *, size_t, void const *, size_t, Realm::Processor);

} // namespace FlexFlow

#endif
