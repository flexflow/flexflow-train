#include "utils/containers/generate_map.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;
using F = std::function<V(K const &)>;

template std::map<K, V> generate_map(std::vector<K> const &, F &&);

} // namespace FlexFlow
