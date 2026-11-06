#include "utils/bidict/algorithms/bidict_from_unordered_map.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template bidict<K, V>
    bidict_from_unordered_map(std::unordered_map<K, V> const &);

} // namespace FlexFlow
