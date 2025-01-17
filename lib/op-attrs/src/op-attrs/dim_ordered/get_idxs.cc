#include "op-attrs/dim_ordered/get_idxs.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  std::vector<ff_dim_t> get_idxs(FFOrdered<T> const &);

} // namespace FlexFlow
