#include "utils/unordered_bidict/algorithms/values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template std::unordered_set<V> values(unordered_bidict<K, V> const &);

} // namespace FlexFlow
