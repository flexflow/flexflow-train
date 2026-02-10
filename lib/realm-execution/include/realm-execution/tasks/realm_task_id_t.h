#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_TASK_ID_T_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_TASK_ID_T_H

#include "realm-execution/realm.h"
#include "realm-execution/tasks/task_id_t.dtg.h"

namespace FlexFlow {

Realm::Processor::TaskFuncID get_realm_task_id_for_task_id(task_id_t);

} // namespace FlexFlow

#endif
