#include "utils/containers/zip_values_strict_with.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V1 = value_type<1>;
using V2 = value_type<2>;
using Out = value_type<3>;
using F = std::function<Out(V1 const &, V2 const &)>;

template std::unordered_map<K, Out> zip_values_strict_with(
    std::unordered_map<K, V1> const &, std::unordered_map<K, V2> const &, F &&);

} // namespace FlexFlow
