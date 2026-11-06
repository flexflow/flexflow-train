#include "utils/unordered_bidict/algorithms/unordered_bidict_unordered_set_of.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

std::unordered_set<std::pair<K, V>>
    unordered_bidict_unordered_set_of(unordered_bidict<K, V> const &);

} // namespace FlexFlow
