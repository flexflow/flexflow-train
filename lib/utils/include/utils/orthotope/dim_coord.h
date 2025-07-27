#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_COORD_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_COORD_H

#include "utils/containers/zip_with_strict.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/orthotope.h"
#include "utils/exception.h"
#include "utils/containers/keys.h"
#include "utils/containers/restrict_keys.h"
#include "utils/exception.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"
#include "utils/containers/scanr.h"
#include "utils/containers/product.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/orthotope/dim_domain.h"

namespace FlexFlow {

template <typename T>
std::unordered_set<T> get_coord_dims(DimCoord<T> const &coord) {
  return keys(coord.raw);  
}

template <typename T>
DimCoord<T> restrict_coord_to_dims(DimCoord<T> const &coord, std::unordered_set<T> const &dims) {
  return DimCoord<T>{
    restrict_keys(coord.raw, dims), 
  };
}

template <typename T>
OrthotopeCoord orthotope_coord_from_dim_coord(DimCoord<T> const &coord) {
  return OrthotopeCoord{
    transform(sorted(get_coord_dims(coord)), [&](T const &t) { return coord.raw.at(t); }),
  };
}

template <typename T>
DimCoord<T> dim_coord_from_orthotope_coord(OrthotopeCoord const &coord, DimDomain<T> const &domain) {
  return DimCoord<T>{
    map_from_keys_and_values(sorted(get_domain_dims(domain)), coord.raw),
  };
}

template <typename T>
nonnegative_int flatten_dim_coord(DimCoord<T> const &coord, 
                              DimDomain<T> const &domain) {
  ASSERT(get_coord_dims(coord) == get_domain_dims(domain),
         "flatten_dim_coord expected coord dimensions to match domain dimensions", 
         coord, domain);

  OrthotopeCoord orthotope_coord = orthotope_coord_from_dim_coord(coord);
  Orthotope orthotope_domain = orthotope_from_dim_domain(domain);

  return flatten_orthotope_coord(orthotope_coord, orthotope_domain);
}

template <typename T>
DimCoord<T> unflatten_dim_coord(nonnegative_int flattened, DimDomain<T> const &domain) {
  Orthotope orthotope_domain = orthotope_from_dim_domain(domain);
  OrthotopeCoord orthotope_coord = unflatten_orthotope_coord(flattened, orthotope_domain);

  return dim_coord_from_orthotope_coord(orthotope_coord, domain);
}

} // namespace FlexFlow

#endif
