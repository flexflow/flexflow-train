#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_MINIMAL_DIM_DOMAIN_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_MINIMAL_DIM_DOMAIN_H

#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/dim_ordering.dtg.h"
#include "utils/orthotope/minimal_orthotope.dtg.h"
#include "utils/orthotope/minimal_dim_domain.dtg.h"
#include "utils/containers/map_values.h"
#include "utils/containers/filtermap_values.h"
#include "utils/containers/keys.h"
#include "utils/containers/transform.h"
#include "utils/containers/sorted_by.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/containers/restrict_keys.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

template <typename T>
MinimalDimDomain<T> empty_minimal_dim_domain() {
  return MinimalDimDomain<T>{{}};
}

template <typename T>
nonnegative_int minimal_dim_domain_num_dims(MinimalDimDomain<T> const &domain) {
  return num_elements(domain.dims);
}

template <typename T>
DimDomain<T> lift_minimal_dim_domain(MinimalDimDomain<T> const &minimal_dim_domain) {
  return DimDomain{
    map_values(minimal_dim_domain.dims,
               [](int_ge_two component) {
                 return component.positive_int_from_int_ge_two();
               }),
  };
}

template <typename T>
MinimalDimDomain<T> require_dim_domain_is_minimal(DimDomain<T> const &dim_domain) {
  return MinimalDimDomain<T>{
    map_values(dim_domain.dims, 
               [](positive_int dim_size) {
                 return int_ge_two{dim_size};
               }),
  };
}

template <typename T>
MinimalDimDomain<T> minimal_dim_domain_from_dim_domain(DimDomain<T> const &dim_domain) {
  return MinimalDimDomain<T>{
    filtermap_values(dim_domain.dims, try_int_ge_two_from_positive_int)
  };
}

template <typename T>
std::unordered_set<T> get_minimal_domain_dims(MinimalDimDomain<T> const &domain) {
  return keys(domain.dims);
}

template <typename T>
MinimalDimDomain<T> restrict_minimal_domain_to_dims(MinimalDimDomain<T> const &domain,
                                                    std::unordered_set<T> const &allowed) {
  return MinimalDimDomain<T>{restrict_keys(domain.dims, allowed)};
}


template <typename T>
MinimalOrthotope minimal_orthotope_from_minimal_dim_domain(
  MinimalDimDomain<T> const &domain,
  DimOrdering<T> const &dim_ordering) {
  
  return MinimalOrthotope{
      transform(sorted_by(get_minimal_domain_dims(domain), dim_ordering.lt),
                [&](T const &t) { return domain.dims.at(t); }),
  };
}

template <typename T>
MinimalDimDomain<T> minimal_dim_domain_from_minimal_orthotope(
  MinimalOrthotope const &orthotope,
  std::unordered_set<T> const &dims,
  DimOrdering<T> const &dim_ordering) {
  
  return MinimalDimDomain<T>{
      map_from_keys_and_values(
          sorted_by(dims, dim_ordering.lt), orthotope.dims),
  };
}

} // namespace FlexFlow

#endif
