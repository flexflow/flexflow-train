#include "utils/ord/unordered_map.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template bool operator<(std::unordered_map<K, V> const &,
                        std::unordered_map<K, V> const &);

} // namespace FlexFlow
