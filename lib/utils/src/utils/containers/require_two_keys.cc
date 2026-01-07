#include "utils/containers/require_two_keys.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template std::pair<V, V>
    require_two_keys(std::unordered_map<K, V> const &, K const &, K const &);

} // namespace FlexFlow
