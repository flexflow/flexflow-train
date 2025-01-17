#include "utils/orthotope/down_projection.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;

template
  DownProjection<T1, T3> compose_down_projections(DownProjection<T1, T2> const &, DownProjection<T2, T3> const &);

using L = value_type<0>;
using R = value_type<1>;

template
  DownProjection<L, R> make_empty_down_projection();

template
  void project_dims(DownProjection<L, R> &, std::unordered_set<L> const &, R const &);

template
  UpProjection<R, L> invert_down_projection(DownProjection<L, R> const &);

template
  DownProjection<L, R> down_from_eq_proj(EqProjection<L, R> const &);

} // namespace FlexFlow
