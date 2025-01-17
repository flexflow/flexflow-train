#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_UP_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_UP_PROJECTION_H

#include "utils/orthotope/up_projection.dtg.h"
#include "utils/orthotope/eq_projection.dtg.h"
#include "utils/orthotope/down_projection.dtg.h"
#include "utils/one_to_many/one_to_many_from_bidict.h"
#include "utils/one_to_many/exhaustive_relational_join.h"
#include "utils/one_to_many/invert_one_to_many.h"

namespace FlexFlow {

template <typename L, typename R>
UpProjection<L, R> make_empty_up_projection() {
  return UpProjection<L, R>{OneToMany<L, R>{}};
}

template <typename L, typename R>
void project_dims(UpProjection<L, R> &proj, L const &onto, std::unordered_set<R> const &from) {
  for (R const &r : from) {
    proj.dim_mapping.insert({onto, r});
  }
}

template <typename L, typename R>
DownProjection<R, L> invert_up_projection(UpProjection<L, R> const &up_proj) {
  return DownProjection<R, L>{
    invert_one_to_many(up_proj.dim_mapping),
  };
}

template <typename T1, typename T2, typename T3>
UpProjection<T1, T3> compose_up_projections(UpProjection<T1, T2> const &fst, UpProjection<T2, T3> const &snd) {
  return UpProjection<T1, T3>{
    exhaustive_relational_join(fst.dim_mapping, snd.dim_mapping),
  };
}

template <typename L, typename R>
UpProjection<L, R> up_from_eq_proj(EqProjection<L, R> const &eq) {
  return UpProjection<L, R>{
    one_to_many_from_bidict(eq.dim_mapping),
  };
}

} // namespace FlexFlow

#endif
