#include "utils/unordered_bidict/algorithms/unordered_bidict_filter_values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using F = std::function<bool(V)>;

template unordered_bidict<K, V> unordered_bidict_filter_values(unordered_bidict<K, V> const &, F &&);

} // namespace FlexFlow
