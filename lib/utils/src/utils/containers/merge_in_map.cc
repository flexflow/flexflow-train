#include "utils/containers/merge_in_map.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;
using V = ordered_value_type<1>;

template void merge_in_map(std::map<K, V> const &,
                           std::map<K, V> &);

} // namespace FlexFlow
