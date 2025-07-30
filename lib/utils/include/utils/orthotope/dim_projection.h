#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_PROJECTION_H

#include "utils/orthotope/dim_coord.h"
#include "utils/orthotope/dim_projection.dtg.h"
#include "utils/orthotope/down_projection.h"
#include "utils/orthotope/eq_projection.h"
#include "utils/orthotope/up_projection.h"
#include "utils/overload.h"

namespace FlexFlow {

template <typename L, typename R>
std::unordered_set<L>
    input_dims_of_projection(DimProjection<L, R> const &projection) {
  return projection.template visit<std::unordered_set<L>>(overload{
      [](UpProjection<L, R> const &p) {
        return input_dims_of_up_projection(p);
      },
      [](EqProjection<L, R> const &p) {
        return input_dims_of_eq_projection(p);
      },
      [](DownProjection<L, R> const &p) {
        return input_dims_of_down_projection(p);
      },
  });
}

template <typename L, typename R>
std::unordered_set<R>
    output_dims_of_projection(DimProjection<L, R> const &projection) {
  return projection.template visit<std::unordered_set<R>>(overload{
      [](UpProjection<L, R> const &p) {
        return output_dims_of_up_projection(p);
      },
      [](EqProjection<L, R> const &p) {
        return output_dims_of_eq_projection(p);
      },
      [](DownProjection<L, R> const &p) {
        return output_dims_of_down_projection(p);
      },
  });
};

template <typename L, typename R>
DimCoord<R> compute_projection(DimProjection<L, R> const &projection,
    DimCoord<L> const &input_coord,
    DimDomain<L> const &input_domain,
    DimDomain<R> const &output_domain,
    DimOrdering<L> const &input_dim_ordering,
    DimOrdering<R> const &output_dim_ordering) {
  ASSERT(dim_domain_contains_coord(input_domain, input_coord));
  ASSERT(get_domain_dims(input_domain) == input_dims_of_projection(projection));
  ASSERT(get_domain_dims(output_domain) == output_dims_of_projection(projection));

  DimCoord<R> output_coord = projection.template visit<DimCoord<R>>(overload{
    [&](UpProjection<L, R> const &p) -> DimCoord<R> {
      return compute_up_projection(p, input_coord, output_domain, output_dim_ordering);
    },
    [&](EqProjection<L, R> const &p) -> DimCoord<R> {
      return compute_eq_projection(p, input_coord);
    },
    [&](DownProjection<L, R> const &p) -> DimCoord<R> {
      return compute_down_projection(p, input_coord, input_domain, input_dim_ordering);
    },
  });

  ASSERT(dim_domain_contains_coord(output_domain, output_coord));

  return output_coord;
}

} // namespace FlexFlow

#endif
