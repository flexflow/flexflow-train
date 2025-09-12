#include "utils/orthotope/dim_coord.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template std::unordered_set<T> get_coord_dims(DimCoord<T> const &);

template DimCoord<T> restrict_coord_to_dims(DimCoord<T> const &,
                                            std::unordered_set<T> const &);

template OrthotopeCoord orthotope_coord_from_dim_coord(DimCoord<T> const &,
                                                       DimOrdering<T> const &);

template DimCoord<T> dim_coord_from_orthotope_coord(OrthotopeCoord const &,
                                                    std::unordered_set<T> const &,
                                                    DimOrdering<T> const &);

template DimCoord<T> lift_dim_coord(DimCoord<T> const &, std::unordered_set<T> const &);

template std::unordered_set<DimCoord<T>> get_coords_in_dim_domain(DimDomain<T> const &);

template std::unordered_set<DimCoord<T>> get_coords_in_minimal_dim_domain(MinimalDimDomain<T> const &);

template DimCoord<T> get_maximum_coord_in_domain(DimDomain<T> const &);

template DimDomain<T> get_domain_for_maximum_coord(DimCoord<T> const &);

template
  bool dim_domain_contains_coord(DimDomain<T> const &,
                                 DimCoord<T> const &);

template
  bool minimal_dim_domain_contains_coord(MinimalDimDomain<T> const &,
                                         DimCoord<T> const &);

template nonnegative_int flatten_dim_coord(DimCoord<T> const &,
                                           DimDomain<T> const &,
                                           DimOrdering<T> const &);

template DimCoord<T> unflatten_dim_coord(nonnegative_int,
                                         DimDomain<T> const &,
                                         DimOrdering<T> const &);

} // namespace FlexFlow
