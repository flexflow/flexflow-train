#include "realm-execution/tasks/serializer/serializable_realm_instance.h"

namespace FlexFlow {

SerializableRealmInstance
    realm_instance_to_serializable(Realm::RegionInstance const &inst) {
  return SerializableRealmInstance{inst.id};
}

Realm::RegionInstance
    realm_instance_from_serializable(SerializableRealmInstance const &inst) {
  return Realm::RegionInstance{inst.id};
}

} // namespace FlexFlow
