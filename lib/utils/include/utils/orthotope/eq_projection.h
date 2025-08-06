#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_EQ_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_EQ_PROJECTION_H

#include "utils/bidict/algorithms/exhaustive_relational_join.h"
#include "utils/containers/map_keys.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/eq_projection.dtg.h"

namespace FlexFlow {

template <typename L, typename R>
EqProjection<L, R> make_empty_eq_projection() {
  return EqProjection<L, R>{bidict<L, R>{}};
}

template <typename L, typename R>
std::unordered_set<L>
    input_dims_of_eq_projection(EqProjection<L, R> const &projection) {
  return projection.dim_mapping.left_values();
}

template <typename L, typename R>
std::unordered_set<R>
    output_dims_of_eq_projection(EqProjection<L, R> const &projection) {
  return projection.dim_mapping.right_values();
}

template <typename L, typename R>
void project_dims(EqProjection<L, R> &proj,
                  L const &from,
                  R const &to) {
  proj.dim_mapping.equate(from, to);
}

template <typename L, typename R>
EqProjection<R, L> invert_eq_projection(EqProjection<L, R> const &input) {
  return EqProjection<R, L>{
      input.dim_mapping.reversed(),
  };
}

template <typename T1, typename T2, typename T3>
EqProjection<T1, T3> compose_eq_projections(EqProjection<T1, T2> const &fst,
                                            EqProjection<T2, T3> const &snd) {
  return EqProjection{
      exhaustive_relational_join(fst.dim_mapping, snd.dim_mapping)};
}

template <typename L, typename R>
DimCoord<R> compute_eq_projection(EqProjection<L, R> const &projection,
                                  DimCoord<L> const &coord) {
  return DimCoord<R>{
      map_keys(coord.raw,
               [&](L const &input_dim) -> R {
                 return projection.dim_mapping.at_l(input_dim);
               }),
  };
};

} // namespace FlexFlow

#endif
