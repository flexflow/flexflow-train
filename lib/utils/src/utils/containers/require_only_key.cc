#include "utils/containers/require_only_key.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template V require_only_key(std::unordered_map<K, V> const &, K const &);

using K2 = ordered_value_type<0>;

template V require_only_key(std::map<K2, V> const &, K2 const &);

} // namespace FlexFlow
