#ifndef _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_SERIALIZABLE_REALM_PROCESSOR_H
#define _FLEXFLOW_LIB_REALM_EXECUTION_INCLUDE_REALM_EXECUTION_TASKS_SERIALIZER_SERIALIZABLE_REALM_PROCESSOR_H

#include "realm-execution/realm.h"
#include "realm-execution/tasks/serializer/serializable_realm_processor.dtg.h"

namespace FlexFlow {

SerializableRealmProcessor
    realm_processor_to_serializable(Realm::Processor const &);
Realm::Processor
    realm_processor_from_serializable(SerializableRealmProcessor const &);

} // namespace FlexFlow

#endif
