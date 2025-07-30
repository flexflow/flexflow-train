#include "utils/orthotope/dim_projection.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template std::unordered_set<L>
    input_dims_of_projection(DimProjection<L, R> const &);

template std::unordered_set<R>
    output_dims_of_projection(DimProjection<L, R> const &);

template 
  DimCoord<R> compute_projection(DimProjection<L, R> const &,
      DimCoord<L> const &,
      DimDomain<L> const &,
      DimDomain<R> const &,
      DimOrdering<L> const &,
      DimOrdering<R> const &);

} // namespace FlexFlow
