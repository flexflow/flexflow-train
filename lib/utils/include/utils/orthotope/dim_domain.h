#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_DOMAIN_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_DOMAIN_H

#include "utils/containers/keys.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/sorted_by.h"
#include "utils/containers/transform.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/dim_ordering.dtg.h"
#include "utils/orthotope/orthotope.dtg.h"
#include "utils/containers/map_from_keys_and_values.h"
#include "utils/nonnegative_int/num_elements.h"

namespace FlexFlow {

template <typename T>
DimDomain<T> empty_dim_domain() {
  return DimDomain<T>{{}};
};

template <typename T>
nonnegative_int dim_domain_num_dims(DimDomain<T> const &domain) {
  return num_elements(domain.dims);
}

template <typename T>
std::unordered_set<T> get_domain_dims(DimDomain<T> const &domain) {
  return keys(domain.dims);
}

template <typename T>
DimDomain<T> restrict_domain_to_dims(DimDomain<T> const &domain,
                                     std::unordered_set<T> const &allowed) {
  return DimDomain<T>{restrict_keys(domain.dims, allowed)};
}

template <typename T>
Orthotope orthotope_from_dim_domain(DimDomain<T> const &domain,
                                    DimOrdering<T> const &dim_ordering) {
  return Orthotope{
      transform(sorted_by(get_domain_dims(domain), dim_ordering.lt),
                [&](T const &t) { return domain.dims.at(t); }),
  };
}

template <typename T>
DimDomain<T> dim_domain_from_orthotope(Orthotope const &orthotope,
                                       std::unordered_set<T> const &dims,
                                       DimOrdering<T> const &dim_ordering) {
  return DimDomain<T>{
      map_from_keys_and_values(
          sorted_by(dims, dim_ordering.lt), orthotope.dims),
  };
}

} // namespace FlexFlow

#endif
