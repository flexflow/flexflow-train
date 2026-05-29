#include "utils/containers/map_values.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = value_type<0>;
using V = value_type<1>;
using V2 = value_type<2>;
using F = std::function<V2(V)>;

template std::unordered_map<K, V2> map_values(std::unordered_map<K, V> const &,
                                              F &&);

using KO = ordered_value_type<0>;
using VO = value_type<1>;
using VO2 = value_type<2>;
using FO = std::function<VO2(VO)>;

template std::map<KO, VO2> map_values(std::map<KO, VO> const &, FO &&);

} // namespace FlexFlow
