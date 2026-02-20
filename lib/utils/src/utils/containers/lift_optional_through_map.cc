#include "utils/containers/lift_optional_through_map.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template std::optional<std::unordered_map<K, V>>
    lift_optional_through_map(std::unordered_map<K, std::optional<V>> const &);

} // namespace FlexFlow
