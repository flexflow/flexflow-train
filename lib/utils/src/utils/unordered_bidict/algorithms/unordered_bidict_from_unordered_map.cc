#include "utils/unordered_bidict/algorithms/unordered_bidict_from_unordered_map.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template unordered_bidict<K, V>
    unordered_bidict_from_unordered_map(std::unordered_map<K, V> const &);

} // namespace FlexFlow
