#include "utils/containers/merge_maps_with_right_dominating.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = value_type<1>;
using C = std::vector<std::map<K, V>>;

template std::map<K, V> merge_maps_with_right_dominating(C const &);

} // namespace FlexFlow
