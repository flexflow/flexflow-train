#include "realm-execution/tasks/serializer/serializable_realm_processor.h"

namespace FlexFlow {

SerializableRealmProcessor
    realm_processor_to_serializable(Realm::Processor const &proc) {
  return SerializableRealmProcessor{proc.id};
}

Realm::Processor
    realm_processor_from_serializable(SerializableRealmProcessor const &proc) {
  return Realm::Processor{proc.id};
}

} // namespace FlexFlow
