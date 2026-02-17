#include "realm-execution/tasks/serializer/serializable_tensor_instance_backing.h"
#include "realm-execution/tasks/serializer/serializable_realm_event.h"
#include "realm-execution/tasks/serializer/serializable_realm_instance.h"
#include "task-spec/dynamic_graph/serializable_dynamic_value_attrs.h"
#include "utils/containers/map_keys_and_values.h"

namespace FlexFlow {

SerializableTensorInstanceBacking tensor_instance_backing_to_serializable(
    TensorInstanceBacking const &backing) {
  return SerializableTensorInstanceBacking{/*backing=*/map_keys_and_values(
      backing.backing,
      dynamic_value_attrs_to_serializable,
      [](std::pair<Realm::RegionInstance, Realm::Event> const &p) {
        return std::pair{realm_instance_to_serializable(p.first),
                         realm_event_to_serializable(p.second)};
      })};
}

TensorInstanceBacking tensor_instance_backing_from_serializable(
    SerializableTensorInstanceBacking const &backing) {
  return TensorInstanceBacking{/*backing=*/map_keys_and_values(
      backing.backing,
      dynamic_value_attrs_from_serializable,
      [](std::pair<SerializableRealmInstance, SerializableRealmEvent> const
             &p) {
        return std::pair{realm_instance_from_serializable(p.first),
                         realm_event_from_serializable(p.second)};
      })};
}

} // namespace FlexFlow
