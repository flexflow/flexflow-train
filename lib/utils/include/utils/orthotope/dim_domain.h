#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_DOMAIN_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_DOMAIN_H

#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/orthotope.dtg.h"
#include "utils/containers/keys.h"
#include "utils/containers/restrict_keys.h"
#include "utils/containers/sorted.h"
#include "utils/containers/transform.h"

namespace FlexFlow {

template <typename T>
std::unordered_set<T> get_domain_dims(DimDomain<T> const &domain) {
  return keys(domain.dims);
}

template <typename T>
DimDomain<T> restrict_domain_to_dims(DimDomain<T> const &domain, std::unordered_set<T> const &allowed) {
  return DimDomain<T>{restrict_keys(domain.dims, allowed)};
}

template <typename T>
Orthotope orthotope_from_dim_domain(DimDomain<T> const &domain) {
  return Orthotope{
    transform(sorted(get_domain_dims(domain)), [&](T const &t) { return domain.dims.at(t); }),
  };
}

} // namespace FlexFlow

#endif
