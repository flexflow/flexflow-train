#include "utils/bidict/algorithms/unordered_set_of.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

std::unordered_set<std::pair<K, V>> unordered_set_of(bidict<K, V> const &);

} // namespace FlexFlow
