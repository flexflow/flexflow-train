#include "kernels/legion_ordered/slice.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  LegionOrdered<T> slice(LegionOrdered<T> const &, legion_dim_t const &, std::optional<legion_dim_t> const &);

} // namespace FlexFlow
