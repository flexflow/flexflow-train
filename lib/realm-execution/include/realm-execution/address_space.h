#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_ADDRESS_SPACE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_ADDRESS_SPACE_H

#include "pcg/node_idx_t.dtg.h"
#include "realm-execution/realm.h"

namespace FlexFlow {

node_idx_t node_idx_from_realm_address_space(Realm::AddressSpace);
Realm::AddressSpace realm_address_space_from_node_idx(node_idx_t);

} // namespace FlexFlow

#endif
