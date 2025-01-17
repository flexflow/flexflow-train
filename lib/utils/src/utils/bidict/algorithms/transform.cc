#include "utils/bidict/algorithms/transform.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using K2 = value_type<2>;
using V2 = value_type<3>;
using F = std::function<std::pair<K2, V2>(K, V)>;

template
  bidict<K2, V2> transform(bidict<K, V> const &, F &&);

} // namespace FlexFlow
