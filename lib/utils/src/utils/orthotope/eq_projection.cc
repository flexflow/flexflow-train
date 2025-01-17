#include "utils/orthotope/eq_projection.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  EqProjection<R, L> invert_eq_projection(EqProjection<L, R> const &);

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;

template
  EqProjection<T1, T3> compose_eq_projections(EqProjection<T1, T2> const &, EqProjection<T2, T3> const &);

} // namespace FlexFlow
