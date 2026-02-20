#include "utils/hash/map.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

using K = ::FlexFlow::ordered_value_type<0>;
using V = ::FlexFlow::value_type<1>;

namespace std {

template struct hash<std::map<K, V>>;

} // namespace std
