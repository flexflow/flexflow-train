#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_SERIALIZABLE_REALM_EVENT_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_SERIALIZABLE_REALM_EVENT_H

#include "realm-execution/realm.h"
#include "realm-execution/tasks/serializer/serializable_realm_event.dtg.h"

namespace FlexFlow {

SerializableRealmEvent realm_event_to_serializable(Realm::Event const &);
Realm::Event realm_event_from_serializable(SerializableRealmEvent const &);

} // namespace FlexFlow

#endif
