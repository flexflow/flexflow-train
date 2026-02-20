#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_UP_PROJECTION_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_UP_PROJECTION_H

#include "utils/containers/flatmap.h"
#include "utils/containers/is_subseteq_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/values.h"
#include "utils/one_to_many/exhaustive_relational_join.h"
#include "utils/one_to_many/invert_one_to_many.h"
#include "utils/one_to_many/one_to_many_from_bidict.h"
#include "utils/orthotope/dim_coord.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/down_projection.dtg.h"
#include "utils/orthotope/eq_projection.dtg.h"
#include "utils/orthotope/up_projection.dtg.h"

namespace FlexFlow {

template <typename L, typename R>
UpProjection<L, R> make_empty_up_projection() {
  return UpProjection<L, R>{OneToMany<L, R>{}};
}

template <typename L, typename R>
std::unordered_set<L>
    input_dims_of_up_projection(UpProjection<L, R> const &projection) {
  return projection.dim_mapping.left_values();
}

template <typename L, typename R>
std::unordered_set<R>
    output_dims_of_up_projection(UpProjection<L, R> const &projection) {
  return projection.dim_mapping.right_values();
}

template <typename L, typename R>
DimCoord<R> compute_up_projection(UpProjection<L, R> const &projection,
                                  DimCoord<L> const &coord,
                                  DimDomain<R> const &output_domain,
                                  DimOrdering<R> const &output_dim_ordering) {
  std::unordered_set<L> input_dims = input_dims_of_up_projection(projection);
  std::unordered_set<L> coord_dims = get_coord_dims(coord);
  ASSERT(input_dims == coord_dims,
         "compute_up_projection expected coord dimensions to match projection "
         "input dimensions");

  std::unordered_set<R> output_dims = output_dims_of_up_projection(projection);
  std::unordered_set<R> output_domain_dims = get_domain_dims(output_domain);
  ASSERT(is_subseteq_of(output_dims, output_domain_dims));

  DimCoord<R> unlifted = DimCoord<R>{
      flatmap(coord.raw,
              [&](L const &input_dim, nonnegative_int input_dim_val) {
                std::unordered_set<R> dst_dims =
                    projection.dim_mapping.at_l(input_dim);

                DimDomain<R> dst_domain =
                    restrict_domain_to_dims(output_domain, dst_dims);

                DimCoord<R> dst_coord = unflatten_dim_coord(
                    input_dim_val, dst_domain, output_dim_ordering);

                return dst_coord.raw;
              }),
  };

  return unlifted;
}

template <typename L, typename R>
void project_dims(UpProjection<L, R> &proj,
                  L const &onto,
                  std::unordered_set<R> const &from) {
  ASSERT(from.size() > 0);

  for (R const &r : from) {
    proj.dim_mapping.insert({onto, r});
  }
}

template <typename L, typename R>
DownProjection<R, L> invert_up_projection(UpProjection<L, R> const &up_proj) {
  return DownProjection<R, L>{
      /*dim_mapping=*/invert_one_to_many(up_proj.dim_mapping),
  };
}

template <typename T1, typename T2, typename T3>
UpProjection<T1, T3> compose_up_projections(UpProjection<T1, T2> const &fst,
                                            UpProjection<T2, T3> const &snd) {
  return UpProjection<T1, T3>{
      /*dim_mapping=*/exhaustive_relational_join(fst.dim_mapping,
                                                 snd.dim_mapping),
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
