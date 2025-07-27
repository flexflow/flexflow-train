#include "utils/orthotope/down_projection.h"
#include "utils/archetypes/value_type.h"
#include "utils/containers/generate_vector.h"
#include "utils/containers/set_of.h"
#include "utils/containers/unordered_set_of.h"
#include "utils/nonnegative_int/num_elements.h"
#include "utils/nonnegative_int/range.h"
#include "utils/orthotope/orthotope.h"
#include "utils/orthotope/orthotope_coord.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;

template
  DownProjection<T1, T3> compose_down_projections(DownProjection<T1, T2> const &, DownProjection<T2, T3> const &);

using L = value_type<0>;
using R = value_type<1>;

template
  DownProjection<L, R> make_empty_down_projection();

template
  void project_dims(DownProjection<L, R> &, std::unordered_set<L> const &, R const &);

template
  UpProjection<R, L> invert_down_projection(DownProjection<L, R> const &);

template
  DownProjection<L, R> down_from_eq_proj(EqProjection<L, R> const &);

Orthotope compute_down_projection(DownProjection<nonnegative_int, nonnegative_int> const &projection,
                                  Orthotope const &domain) {
  NOT_IMPLEMENTED();
}

OrthotopeCoord compute_down_projection(DownProjection<nonnegative_int, nonnegative_int> const &projection,
                                       OrthotopeCoord const &coord,
                                       Orthotope const &domain) {
  std::unordered_set<nonnegative_int> input_dims = input_dims_of_down_projection(projection);
  std::unordered_set<nonnegative_int> orthotope_dims = unordered_set_of(range(orthotope_get_num_dims(domain)));

  ASSERT(input_dims == orthotope_dims, 
         "compute_down_projection expected projection input dims to match orthotope dims", 
         input_dims, orthotope_dims);

  std::unordered_set<nonnegative_int> output_dims = output_dims_of_down_projection(projection);

  return OrthotopeCoord{
    generate_vector(
                 num_elements(output_dims),
                 [&](nonnegative_int const &output_dim) -> nonnegative_int {
                   std::unordered_set<nonnegative_int> src_dims = projection.dim_mapping.at_r(output_dim);

                   OrthotopeCoord src_coord = restrict_orthotope_coord_to_dims(coord, set_of(src_dims));
                   Orthotope src_domain = restrict_orthotope_to_dims(domain, set_of(src_dims));

                   return flatten_orthotope_coord(src_coord, src_domain);
                 }),
  };
}
} // namespace FlexFlow
