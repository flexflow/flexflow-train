#include "utils/containers/binary_merge_maps_with_left_dominating.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;

template std::map<K, V>
    binary_merge_maps_with_left_dominating(std::map<K, V> const &,
                                           std::map<K, V> const &);

} // namespace FlexFlow
