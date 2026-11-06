#include "utils/unordered_bidict/algorithms/unordered_bidict_transform.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using K2 = value_type<2>;
using V2 = value_type<3>;
using F = std::function<std::pair<K2, V2>(K, V)>;

template unordered_bidict<K2, V2>
    unordered_bidict_transform(unordered_bidict<K, V> const &, F &&);

} // namespace FlexFlow
