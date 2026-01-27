#include "op-attrs/ff_ordered/filtrans.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using In = value_type<0>;
using Out = value_type<1>;
using F = std::function<std::optional<Out>(In const &)>;

template FFOrdered<Out> filtrans(FFOrdered<In> const &, F &&);

} // namespace FlexFlow
