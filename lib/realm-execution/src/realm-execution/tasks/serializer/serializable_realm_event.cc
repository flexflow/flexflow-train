#include "realm-execution/tasks/serializer/serializable_realm_event.h"

namespace FlexFlow {

SerializableRealmEvent realm_event_to_serializable(Realm::Event const &event) {
  return SerializableRealmEvent{event.id};
}

Realm::Event
    realm_event_from_serializable(SerializableRealmEvent const &event) {
  return Realm::Event{event.id};
}

} // namespace FlexFlow
