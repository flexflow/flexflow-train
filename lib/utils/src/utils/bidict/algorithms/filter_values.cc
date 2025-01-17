#include "utils/bidict/algorithms/filter_values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using F = std::function<bool(V)>;

template
  bidict<K, V> filter_values(bidict<K, V> const &, F &&);

} // namespace FlexFlow
