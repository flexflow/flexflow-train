#include "utils/containers/zip_values_strict.h"
#include "utils/archetypes/value_type.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using VT0 = value_type<0>;
using VT1 = value_type<1>;
using VT2 = value_type<2>;

template std::unordered_map<VT0, std::pair<VT1, VT2>>
    zip_values_strict(std::unordered_map<VT0, VT1> const &,
                      std::unordered_map<VT0, VT2> const &);

using OV0 = ordered_value_type<0>;

template std::map<OV0, std::pair<VT1, VT2>>
    zip_values_strict(std::map<OV0, VT1> const &,
                      std::map<OV0, VT2> const &);



} // namespace FlexFlow
