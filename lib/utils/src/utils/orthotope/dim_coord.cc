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
                                                    DimDomain<T> const &,
                                                    DimOrdering<T> const &);

template nonnegative_int flatten_dim_coord(DimCoord<T> const &,
                                           DimDomain<T> const &,
                                           DimOrdering<T> const &);

template DimCoord<T> unflatten_dim_coord(nonnegative_int,
                                         DimDomain<T> const &,
                                         DimOrdering<T> const &);

} // namespace FlexFlow
