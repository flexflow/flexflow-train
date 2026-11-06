#include "utils/unordered_bidict/algorithms/unordered_bidict_transform_keys_and_values.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using K2 = value_type<2>;
using V2 = value_type<3>;
using KF = std::function<K2(K const &)>;
using VF = std::function<V2(V const &)>;

template unordered_bidict<K2, V2>
    unordered_bidict_transform_keys_and_values(unordered_bidict<K, V> const &, KF &&, VF &&);

} // namespace FlexFlow
