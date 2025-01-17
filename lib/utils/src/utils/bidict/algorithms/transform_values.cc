#include "utils/bidict/algorithms/transform_values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using V2 = value_type<2>;
using F = std::function<V2(V)>;

template
  bidict<K, V2> transform_values(bidict<K, V> const &, F &&);

} // namespace FlexFlow
