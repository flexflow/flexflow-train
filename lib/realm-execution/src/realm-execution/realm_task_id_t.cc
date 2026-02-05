#include "realm-execution/realm_task_id_t.h"

namespace FlexFlow {

Realm::Processor::TaskFuncID get_realm_task_id_for_task_id(task_id_t task_id) {
  return Realm::Processor::TASK_ID_FIRST_AVAILABLE +
         static_cast<Realm::Processor::TaskFuncID>(task_id);
}

} // namespace FlexFlow
