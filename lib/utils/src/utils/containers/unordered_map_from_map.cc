#include "utils/containers/unordered_map_from_map.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<0>;

template std::unordered_map<K, V> unordered_map_from_map(std::map<K, V> const &);

} // namespace FlexFlow
