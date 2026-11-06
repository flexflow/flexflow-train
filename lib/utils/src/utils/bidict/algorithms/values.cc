#include "utils/bidict/algorithms/values.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template std::set<V> values(bidict<K, V> const &);

} // namespace FlexFlow
