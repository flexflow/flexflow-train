#include "utils/containers/map_keys_and_values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using K2 = value_type<2>;
using V2 = value_type<3>;
using FK = std::function<K2(K const &)>;
using FV = std::function<V2(V const &)>;

template std::unordered_map<K2, V2> map_keys_and_values(
    std::unordered_map<K, V> const &, FK const &, FV const &);

} // namespace FlexFlow
