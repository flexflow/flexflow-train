#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_DOMAIN_MAPPING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_DIM_DOMAIN_MAPPING_H

#include "utils/bidict/bidict.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/dim_domain.dtg.h"
#include "utils/orthotope/dim_ordering.dtg.h"
#include "utils/orthotope/dim_projection.h"
#include "utils/orthotope/dim_coord.h"
#include "utils/bidict/algorithms/left_entries.h"
#include "utils/bidict/algorithms/right_entries.h"
#include "utils/bidict/algorithms/exhaustive_relational_join.h"
#include "utils/hash/tuple.h"
#include "utils/bidict/generate_bidict.h"
#include "utils/orthotope/minimal_dim_domain.dtg.h"
#include "utils/orthotope/minimal_dim_domain.dtg.h"

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

  bool operator==(DimDomainMapping<L, R> const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(DimDomainMapping<L, R> const &other) const {
    return this->tie() != other.tie();
  }

public:
  bidict<DimCoord<L>, DimCoord<R>> coord_mapping;
  DimDomain<L> l_domain;
  DimDomain<R> r_domain;

private:
  std::tuple<
    decltype(coord_mapping) const &,
    decltype(l_domain) const &,
    decltype(r_domain) const &
  > tie() const {
    return std::tie(
      this->coord_mapping,
      this->l_domain,
      this->r_domain);
  }

  friend struct ::std::hash<DimDomainMapping<L, R>>;
};

template <typename L, typename R>
std::string format_as(DimDomainMapping<L, R> const &m) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return fmt::format(
    "<DimDomainMapping l_domain={} r_domain={} coord_mapping={}>",
    m.l_domain,
    m.r_domain,
    m.coord_mapping);
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, DimDomainMapping<L, R> const &m) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return (s << fmt::to_string(m));
}

template <typename L, typename R>
DimDomainMapping<L, R> empty_dim_domain_mapping() {
  return DimDomainMapping{
    /*coord_mapping=*/{
      {DimCoord<L>{{}}, DimCoord<R>{{}}},
    },
    /*l_domain=*/empty_dim_domain<L>(),
    /*r_domain=*/empty_dim_domain<R>(),
  };
}

template <typename L, typename R>
DimDomainMapping<L, R> dim_domain_mapping_identity_map(
    DimDomain<L> const &l_domain, 
    DimDomain<R> const &r_domain, 
    DimOrdering<L> const &l_dim_ordering, 
    DimOrdering<R> const &r_dim_ordering) {
  DimProjection<L, R> projection =
    dim_projection_identity_map(l_domain, r_domain, l_dim_ordering, r_dim_ordering);

  return dim_domain_mapping_from_projection(
    /*projection=*/projection,
    /*l_domain=*/l_domain,
    /*r_domain=*/r_domain,
    /*l_dim_ordering=*/l_dim_ordering,
    /*r_dim_ordering=*/r_dim_ordering);
}

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


template <typename L, typename R>
DimDomainMapping<L, R> dim_domain_mapping_from_projection(
  DimProjection<L, R> const &projection,
  DimDomain<L> const &l_domain,
  DimDomain<R> const &r_domain,
  DimOrdering<L> const &l_dim_ordering,
  DimOrdering<R> const &r_dim_ordering) {
  
  return DimDomainMapping{
    /*coord_mapping=*/generate_bidict(
      get_coords_in_dim_domain(l_domain),
      [&](DimCoord<L> const &l_coord) {
        return compute_dim_projection(
          /*projection=*/projection,
          /*input_coord=*/l_coord,
          /*input_domain=*/l_domain,
          /*output_domain=*/r_domain,
          /*input_dim_ordering=*/l_dim_ordering,
          /*output_dim_ordering=*/r_dim_ordering);
      }),
    /*l_domain=*/l_domain,
    /*r_domain=*/r_domain,
  };
}

} // namespace FlexFlow

namespace std {

template <typename L, typename R>
struct hash<::FlexFlow::DimDomainMapping<L, R>> {
  size_t operator()(::FlexFlow::DimDomainMapping<L, R> const &dim_domain_mapping) const {
    return get_std_hash(dim_domain_mapping.tie()); 
  }
};

} // namespace std

#endif
