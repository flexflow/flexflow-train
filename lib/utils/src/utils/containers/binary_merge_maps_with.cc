#include "utils/containers/binary_merge_maps_with.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using F = std::function<V(V const &, V const &)>;

template
  std::unordered_map<K, V> binary_merge_maps_with(
    std::unordered_map<K, V> const &,
    std::unordered_map<K, V> const &,
    F &&);

} // namespace FlexFlow
