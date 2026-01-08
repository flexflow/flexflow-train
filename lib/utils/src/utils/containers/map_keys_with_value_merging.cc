#include "utils/containers/map_keys_with_value_merging.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using K2 = value_type<2>;

using F = std::function<K2(K const &)>;
using MergeF = std::function<V(V const &, V const &)>;

template std::unordered_map<K2, V> map_keys_with_value_merging(
    std::unordered_map<K, V> const &, F &&, MergeF &&);

} // namespace FlexFlow
