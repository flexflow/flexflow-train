#include "utils/containers/generate_unordered_map.h"
#include "utils/archetypes/value_type.h"
#include <unordered_set>

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using F = std::function<V(K const &)>;

template std::unordered_map<K, V> generate_unordered_map(std::unordered_set<K> const &, F &&);

} // namespace FlexFlow
