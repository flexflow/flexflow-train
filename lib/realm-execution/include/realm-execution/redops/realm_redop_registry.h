#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_REDOP_REGISTRY_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_REALM_REDOP_REGISTRY_H

#include "realm-execution/realm.h"
#include "realm-execution/redops/redop_id_t.dtg.h"

namespace FlexFlow {

/**
 * \brief Registers all known reduction operators (redops).
 */
void Realm::Event register_all_redops(Realm::Runtime);

} // namespace FlexFlow

#endif
