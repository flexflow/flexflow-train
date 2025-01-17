#include "utils/orthotope/up_projection.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;

template
  UpProjection<T1, T3> compose_up_projections(UpProjection<T1, T2> const &, UpProjection<T2, T3> const &);

using L = value_type<0>;
using R = value_type<1>;

template
  UpProjection<L, R> make_empty_up_projection();

template
  void project_dims(UpProjection<L, R> &, L const &, std::unordered_set<R> const &);

template
  DownProjection<R, L> invert_up_projection(UpProjection<L, R> const &);

template 
  UpProjection<L, R> up_from_eq_proj(EqProjection<L, R> const &);

} // namespace FlexFlow
