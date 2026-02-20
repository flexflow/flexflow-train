#include "utils/bijection/bijection.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template Bijection<R, L> flip_bijection(Bijection<L, R> const &);

} // namespace FlexFlow
