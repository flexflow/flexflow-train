#include "utils/orthotope/dim_coord.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template
  std::unordered_set<T> get_coord_dims(DimCoord<T> const &);

template
  DimCoord<T> restrict_coord_to_dims(DimCoord<T> const &, std::unordered_set<T> const &);

template
  OrthotopeCoord orthotope_coord_from_dim_coord(DimCoord<T> const &);

template
  nonnegative_int flatten_dim_coord(DimCoord<T> const &, 
                                    DimDomain<T> const &);

template
  DimCoord<T> unflatten_dim_coord(nonnegative_int, 
                                  DimDomain<T> const &);

} // namespace FlexFlow
