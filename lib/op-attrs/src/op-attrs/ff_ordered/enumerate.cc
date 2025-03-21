#include "op-attrs/ff_ordered/enumerate.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template std::map<ff_dim_t, T> enumerate(FFOrdered<T> const &);

} // namespace FlexFlow
