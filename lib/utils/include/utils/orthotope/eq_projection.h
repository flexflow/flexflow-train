#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_EQ_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_EQ_PROJECTION_H

#include "utils/orthotope/eq_projection.dtg.h"
#include "utils/bidict/algorithms/exhaustive_relational_join.h"

namespace FlexFlow {

template <typename L, typename R>
EqProjection<R, L> invert_eq_projection(EqProjection<L, R> const &input) {
  return EqProjection<R, L>{
    input.dim_mapping.reversed(),
  };
}

template <typename T1, typename T2, typename T3>
EqProjection<T1, T3> compose_eq_projections(EqProjection<T1, T2> const &fst, EqProjection<T2, T3> const &snd) {
  return EqProjection{
    exhaustive_relational_join(fst.dim_mapping, snd.dim_mapping)
  };
}


} // namespace FlexFlow

#endif
