#include "utils/bidict/algorithms/bidict_transform.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;
using K2 = ordered_value_type<2>;
using V2 = ordered_value_type<3>;
using F = std::function<std::pair<K2, V2>(K, V)>;

template bidict<K2, V2> bidict_transform(bidict<K, V> const &, F &&);

} // namespace FlexFlow
