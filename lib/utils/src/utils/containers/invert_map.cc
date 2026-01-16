#include "utils/containers/invert_map.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {
using K = value_type<0>;
using V = value_type<1>;

template std::unordered_map<V, std::unordered_set<K>> invert_map(std::unordered_map<K, V> const &);

} // namespace FlexFlow
