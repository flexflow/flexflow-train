#include "utils/containers/values.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template std::multiset<V> values(std::map<K, V> const &);

template std::unordered_multiset<V> values(std::unordered_map<K, V> const &);

} // namespace FlexFlow
