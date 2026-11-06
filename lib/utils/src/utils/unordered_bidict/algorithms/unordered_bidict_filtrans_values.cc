#include "utils/unordered_bidict/algorithms/unordered_bidict_filtrans_values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using V2 = value_type<2>;
using F = std::function<std::optional<V2>(V)>;

template unordered_bidict<K, V2> unordered_bidict_filtrans_values(unordered_bidict<K, V> const &, F &&);

} // namespace FlexFlow
