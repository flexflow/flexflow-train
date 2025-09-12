#include "op-attrs/ff_ordered/map_from_ff_ordered.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  std::unordered_map<ff_dim_t, T> map_from_ff_ordered(FFOrdered<T> const &);

} // namespace FlexFlow
