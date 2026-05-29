#include "utils/containers/map_keys.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using VT0 = value_type<0>;
using VT1 = value_type<1>;
using VT2 = value_type<2>;

template
  std::unordered_map<VT2, VT0> map_keys(std::unordered_map<VT1, VT0> const &,
                                        std::function<VT2(VT1 const &)> &&);

using OV0 = ordered_value_type<0>;
using OV1 = ordered_value_type<1>;

template
  std::map<OV1, VT0> map_keys(std::map<OV0, VT0> const &m,
                              std::function<OV1(OV0 const &)> &&);

} // namespace FlexFlow
