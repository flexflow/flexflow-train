#include "utils/unordered_bidict/algorithms/unordered_bidict_filtrans_keys.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using K2 = value_type<2>;
using F = std::function<std::optional<K2>(K)>;

template unordered_bidict<K2, V> unordered_bidict_filtrans_keys(unordered_bidict<K, V> const &, F &&);

} // namespace FlexFlow
