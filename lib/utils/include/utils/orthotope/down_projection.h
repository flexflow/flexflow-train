#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DOWN_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DOWN_PROJECTION_H

#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/down_projection.dtg.h"
#include "utils/orthotope/eq_projection.dtg.h"
#include "utils/orthotope/orthotope.dtg.h"
#include "utils/orthotope/orthotope.h"
#include "utils/orthotope/orthotope_coord.dtg.h"
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
std::unordered_set<L> input_dims_of_down_projection(DownProjection<L, R> const &projection) {
  return projection.dim_mapping.left_values();
}

template <typename L, typename R>
std::unordered_set<R> output_dims_of_down_projection(DownProjection<L, R> const &projection) {
  return projection.dim_mapping.right_values();
}

template <typename L, typename R>
DimCoord<R> compute_down_projection(DownProjection<L, R> const &projection, 
                                    DimCoord<L> const &coord, 
                                    DimDomain<L> const &domain) {
  std::unordered_set<L> input_dims = input_dims_of_down_projection(projection);
  std::unordered_set<L> coord_dims = get_coord_dims(coord);
  if (input_dims != coord_dims) {
    throw mk_runtime(fmt::format("compute_down_projection expected coord dimensions to match projection input dimensions, but received inputs_dims={} and coord_dims={}", input_dims, coord_dims));
  }

  std::unordered_set<R> output_dims = output_dims_of_down_projection(projection);

  return DimCoord<R>{
    generate_map(output_dims, 
                 [&](R const &output_dim) {
                   std::unordered_set<L> src_dims = projection.dim_mapping.at_r(output_dim);

                   DimCoord<L> src_coord = restrict_coord_to_dims(coord, src_dims);
                   DimDomain<L> src_domain = restrict_domain_to_dims(domain, src_dims);

                   return flatten_coord(src_coord, src_domain);
                 }),
  };
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
