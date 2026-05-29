#include "utils/containers/restrict_keys.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using VT0 = value_type<0>;
using VT1 = value_type<1>;

template
  std::unordered_map<VT0, VT1> restrict_keys(std::unordered_map<VT0, VT1> const &,
                                             std::unordered_set<VT0> const &);

using OV0 = ordered_value_type<0>;

template
  std::map<OV0, VT1> restrict_keys(std::map<OV0, VT1> const &,
                                   std::set<OV0> const &);


} // namespace FlexFlow
