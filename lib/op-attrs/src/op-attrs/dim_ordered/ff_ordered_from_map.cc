#include "op-attrs/dim_ordered/ff_ordered_from_map.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template 
  FFOrdered<T> ff_ordered_from_map(std::map<ff_dim_t, T> const &);

template
  FFOrdered<T> ff_ordered_from_map(std::unordered_map<ff_dim_t, T> const &);

} // namespace FlexFlow
