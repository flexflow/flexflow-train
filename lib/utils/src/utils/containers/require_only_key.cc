#include "utils/containers/require_only_key.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template V require_only_key(std::unordered_map<K, V> const &, K const &);


} // namespace FlexFlow
