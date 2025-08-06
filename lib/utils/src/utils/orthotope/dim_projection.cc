#include "utils/orthotope/dim_projection.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template std::unordered_set<L>
    input_dims_of_projection(DimProjection<L, R> const &);

template std::unordered_set<R>
    output_dims_of_projection(DimProjection<L, R> const &);

template 
  DimProjection<R, L> invert_dim_projection(DimProjection<L, R> const &);

template DimCoord<R> compute_projection(DimProjection<L, R> const &,
                                        DimCoord<L> const &,
                                        DimDomain<L> const &,
                                        DimDomain<R> const &,
                                        DimOrdering<L> const &,
                                        DimOrdering<R> const &);

using T1 = value_type<2>;
using T2 = value_type<3>;
using T3 = value_type<4>;

template DimProjection<T1, T3> right_compose_eq_projection(
   DimProjection<T1, T2> const &,
   EqProjection<T2, T3> const &);

template DimProjection<T1, T3> left_compose_eq_projection(
    EqProjection<T1, T2> const &,
    DimProjection<T2, T3> const &);

template DimProjection<T1, T3> compose_dim_projections(
  DimProjection<T1, T2> const &,
  DimProjection<T2, T3> const &);

} // namespace FlexFlow
