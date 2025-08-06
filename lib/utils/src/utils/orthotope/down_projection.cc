#include "utils/orthotope/down_projection.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  DownProjection<L, R> make_empty_down_projection();

template
  std::unordered_set<L>
      input_dims_of_down_projection(DownProjection<L, R> const &);

template
  std::unordered_set<R>
      output_dims_of_down_projection(DownProjection<L, R> const &);

template
  DimCoord<R> compute_down_projection(DownProjection<L, R> const &,
                                      DimCoord<L> const &,
                                      DimDomain<L> const &,
                                      DimOrdering<L> const &);

template void project_dims(DownProjection<L, R> &,
                           std::unordered_set<L> const &,
                           R const &);

template
  UpProjection<R, L>
      invert_down_projection(DownProjection<L, R> const &);

using T1 = value_type<2>;
using T2 = value_type<3>;
using T3 = value_type<4>;

template
  DownProjection<T1, T3>
      compose_down_projections(DownProjection<T1, T2> const &,
                               DownProjection<T2, T3> const &);

template
  DownProjection<L, R> down_from_eq_proj(EqProjection<L, R> const &);


} // namespace FlexFlow
