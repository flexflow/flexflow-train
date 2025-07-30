#include "utils/orthotope/dim_ordering.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template
  DimOrdering<T> make_default_dim_ordering();

template
  DimOrdering<T> make_reversed_dim_ordering();

} // namespace FlexFlow
