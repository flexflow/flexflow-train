#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_SERIALIZABLE_REALM_INSTANCE_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_SERIALIZABLE_REALM_INSTANCE_H

#include "realm-execution/realm.h"
#include "realm-execution/tasks/serializer/serializable_realm_instance.dtg.h"

namespace FlexFlow {

SerializableRealmInstance
    realm_instance_to_serializable(Realm::RegionInstance const &);
Realm::RegionInstance
    realm_instance_from_serializable(SerializableRealmInstance const &);

} // namespace FlexFlow

#endif
