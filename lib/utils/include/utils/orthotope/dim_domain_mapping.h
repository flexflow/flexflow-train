#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_DOMAIN_MAPPING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_DOMAIN_MAPPING_H

#include "utils/bidict/bidict.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/dim_domain.dtg.h"

namespace FlexFlow {

template <typename L, typename R>
struct DimDomainMapping {
public:
  explicit DimDomainMapping(
    bidict<DimCoord<L>, DimCoord<R>> const &coord_mapping,
    DimDomain<L> const &l_domain,
    DimDomain<R> const &r_domain)
      : coord_mapping(coord_mapping),
        l_domain(l_domain),
        r_domain(r_domain)
    { 
      ASSERT(get_coords_in_dim_domain(l_domain) == left_entries(coord_mapping));
      ASSERT(get_coords_in_dim_domain(r_domain) == right_entries(coord_mapping));
    } 

  DimCoord<R> at_l(DimCoord<L> const &l_coord) const {
    ASSERT(dim_domain_contains_coord(this->l_domain, l_coord));

    return this->coord_mapping.at_l(l_coord);
  }

  DimCoord<L> at_r(DimCoord<R> const &r_coord) const {
    ASSERT(dim_domain_contains_coord(this->r_domain, r_coord));

    return this->coord_mapping.at_r(r_coord);
  }
public:
  bidict<DimCoord<L>, DimCoord<R>> coord_mapping;
  DimDomain<L> l_domain;
  DimDomain<R> r_domain;
};

template <typename L, typename R>
DimDomainMapping<R, L> invert_dim_domain_mapping(
  DimDomainMapping<L, R> const &dim_domain_mapping) {
  
  return DimDomainMapping{
    /*coord_mapping=*/dim_domain_mapping.coord_mapping.reversed(),
    /*l_domain=*/dim_domain_mapping.r_domain,
    /*r_domain=*/dim_domain_mapping.l_domain,
  };
}

template <typename T1, typename T2, typename T3>
DimDomainMapping<T1, T3> compose_dim_domain_mappings(
  DimDomainMapping<T1, T2> const &lhs,
  DimDomainMapping<T2, T3> const &rhs) {
 
  ASSERT(lhs.r_domain == rhs.l_domain);
  
  return DimDomainMapping{
    /*coord_mapping=*/exhaustive_relational_join(
      lhs.coord_mapping,
      rhs.coord_mapping),
    /*l_domain=*/lhs.l_domain,
    /*r_domain=*/rhs.r_domain,
  };
}

} // namespace FlexFlow

#endif
