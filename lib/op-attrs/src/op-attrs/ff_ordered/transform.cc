#include "op-attrs/ff_ordered/transform.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;
using Out = value_type<1>;
using F = std::function<Out(T const &)>;

template FFOrdered<Out> transform(FFOrdered<T> const &, F &&);

} // namespace FlexFlow
