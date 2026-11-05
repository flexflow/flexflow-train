#include "op-attrs/ff_ordered/ff_ordered_filtrans_with_idx.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using In = value_type<0>;
using Out = value_type<1>;
using F = std::function<std::optional<Out>(ff_dim_t const &, In const &)>;

template FFOrdered<Out> ff_ordered_filtrans_with_idx(FFOrdered<In> const &,
                                                     F &&);

} // namespace FlexFlow
