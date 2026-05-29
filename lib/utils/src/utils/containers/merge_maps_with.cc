#include "utils/containers/merge_maps_with.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;
using F = std::function<V(V const &, V const &)>;

template std::map<K, V>
    merge_maps_with(std::vector<std::map<K, V>> const &, F &&);

} // namespace FlexFlow
