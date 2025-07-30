#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_COORD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_COORD_H

#include "utils/containers/all_of.h"
#include "utils/containers/keys.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/containers/product.h"
#include "utils/containers/require_same.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/scanr.h"
#include "utils/containers/sorted_by.h"
#include "utils/containers/transform.h"
#include "utils/containers/zip_with_strict.h"
#include "utils/exception.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/dim_domain.h"
#include "utils/orthotope/orthotope.h"

namespace FlexFlow {

template <typename T>
std::unordered_set<T> get_coord_dims(DimCoord<T> const &coord) {
  return keys(coord.raw);
}

template <typename T>
DimCoord<T> restrict_coord_to_dims(DimCoord<T> const &coord,
                                   std::unordered_set<T> const &dims) {
  return DimCoord<T>{
      restrict_keys(coord.raw, dims),
  };
}

template <typename T>
OrthotopeCoord
    orthotope_coord_from_dim_coord(DimCoord<T> const &coord,
                                   DimOrdering<T> const &dim_ordering) {
  return OrthotopeCoord{
      transform(sorted_by(get_coord_dims(coord), dim_ordering.lt),
                [&](T const &t) { return coord.raw.at(t); }),
  };
}

template <typename T>
DimCoord<T> dim_coord_from_orthotope_coord(OrthotopeCoord const &coord,
                                           DimDomain<T> const &domain,
                                           DimOrdering<T> const &dim_ordering) {
  return DimCoord<T>{
      map_from_keys_and_values(
          sorted_by(get_domain_dims(domain), dim_ordering.lt), coord.raw),
  };
}

template <typename T>
bool dim_domain_contains_coord(DimDomain<T> const &domain,
                               DimCoord<T> const &coord) {
  ASSERT(get_domain_dims(domain) == get_coord_dims(coord));

  std::unordered_set<T> dims =
      require_same(get_domain_dims(domain), get_coord_dims(coord));
  return all_of(dims, [&](T const &dim) {
    return coord.raw.at(dim) < domain.dims.at(dim);
  });
}

template <typename T>
nonnegative_int flatten_dim_coord(DimCoord<T> const &coord,
                                  DimDomain<T> const &domain,
                                  DimOrdering<T> const &dim_ordering) {
  ASSERT(
      get_coord_dims(coord) == get_domain_dims(domain),
      "flatten_dim_coord expected coord dimensions to match domain dimensions",
      coord,
      domain);

  OrthotopeCoord orthotope_coord =
      orthotope_coord_from_dim_coord(coord, dim_ordering);
  Orthotope orthotope_domain = orthotope_from_dim_domain(domain, dim_ordering);

  return flatten_orthotope_coord(orthotope_coord, orthotope_domain);
}

template <typename T>
DimCoord<T> unflatten_dim_coord(nonnegative_int flattened,
                                DimDomain<T> const &domain,
                                DimOrdering<T> const &dim_ordering) {
  Orthotope orthotope_domain = orthotope_from_dim_domain(domain, dim_ordering);
  OrthotopeCoord orthotope_coord =
      unflatten_orthotope_coord(flattened, orthotope_domain);

  return dim_coord_from_orthotope_coord(orthotope_coord, domain, dim_ordering);
}

} // namespace FlexFlow

#endif
