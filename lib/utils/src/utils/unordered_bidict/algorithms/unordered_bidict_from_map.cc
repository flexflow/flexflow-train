#include "utils/unordered_bidict/algorithms/unordered_bidict_from_map.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template unordered_bidict<K, V> unordered_bidict_from_map(std::map<K, V> const &);

template unordered_bidict<K, V> unordered_bidict_from_map(std::unordered_map<K, V> const &);

} // namespace FlexFlow
