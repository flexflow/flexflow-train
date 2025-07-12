#include "utils/containers/contains_value.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;

template bool contains_value(std::unordered_map<K, V> const &, V const &);

template bool contains_value(std::map<K, V> const &, V const &);

} // namespace FlexFlow
