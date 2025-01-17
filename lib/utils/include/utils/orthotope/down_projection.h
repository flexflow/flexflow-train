#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DOWN_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DOWN_PROJECTION_H

#include "utils/orthotope/down_projection.dtg.h"
#include "utils/orthotope/eq_projection.dtg.h"
#include "utils/orthotope/up_projection.dtg.h"
#include "utils/many_to_one/many_to_one_from_bidict.h"
#include "utils/many_to_one/exhaustive_relational_join.h"
#include "utils/many_to_one/invert_many_to_one.h"

namespace FlexFlow {

template <typename L, typename R>
DownProjection<L, R> make_empty_down_projection() {
  return DownProjection<L, R>{ManyToOne<L, R>{}};
}

template <typename L, typename R>
void project_dims(DownProjection<L, R> &proj, std::unordered_set<L> const &from, R const &onto) {
  for (L const &l : from) {
    proj.dim_mapping.insert({l, onto});
  }
}

template <typename L, typename R>
UpProjection<R, L> invert_down_projection(DownProjection<L, R> const &down_proj) {
  return UpProjection<R, L>{
    invert_many_to_one(down_proj.dim_mapping),
  };
}

template <typename T1, typename T2, typename T3>
DownProjection<T1, T3> compose_down_projections(DownProjection<T1, T2> const &fst, DownProjection<T2, T3> const &snd) {
  return DownProjection<T1, T3>{
    exhaustive_relational_join(fst.dim_mapping, snd.dim_mapping),
  };
}

template <typename L, typename R>
DownProjection<L, R> down_from_eq_proj(EqProjection<L, R> const &eq) {
  return DownProjection<L, R>{
    many_to_one_from_bidict(eq.dim_mapping),
  };
}


} // namespace FlexFlow

#endif
