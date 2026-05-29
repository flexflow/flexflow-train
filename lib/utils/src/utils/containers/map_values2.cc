#include "utils/containers/map_values2.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using VT0 = value_type<0>;
using VT1 = value_type<1>;
using VT2 = value_type<2>;

template std::unordered_map<VT0, VT2> map_values2(
    std::unordered_map<VT0, VT1> const &,
    std::function<VT2(VT0 const &, VT1 const &)> &&);

using OT0 = ordered_value_type<0>;

template std::map<OT0, VT2> map_values2(
    std::map<OT0, VT1> const &,
    std::function<VT2(OT0 const &, VT1 const &)> &&);

} // namespace FlexFlow
