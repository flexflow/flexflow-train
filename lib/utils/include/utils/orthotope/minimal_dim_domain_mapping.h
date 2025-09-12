#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_MINIMAL_DIM_DOMAIN_MAPPING_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ORTHOTOPE_MINIMAL_DIM_DOMAIN_MAPPING_H

#include "utils/bidict/bidict.h"
#include "utils/orthotope/dim_coord.dtg.h"
#include "utils/orthotope/minimal_dim_domain.dtg.h"
#include "utils/orthotope/dim_domain_mapping.h"
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
#include "utils/bidict/algorithms/transform_values.h"
#include "utils/bidict/algorithms/transform_keys.h"

namespace FlexFlow {

template <typename L, typename R>
struct MinimalDimDomainMapping {
public:
  explicit MinimalDimDomainMapping(
    bidict<DimCoord<L>, DimCoord<R>> const &coord_mapping,
    MinimalDimDomain<L> const &l_domain,
    MinimalDimDomain<R> const &r_domain)
      : coord_mapping(coord_mapping),
        l_domain(l_domain),
        r_domain(r_domain)
    { 
      ASSERT(get_coords_in_minimal_dim_domain(l_domain) == left_entries(coord_mapping));
      ASSERT(get_coords_in_minimal_dim_domain(r_domain) == right_entries(coord_mapping));
    } 

  DimCoord<R> at_l(DimCoord<L> const &l_coord) const {
    ASSERT(minimal_dim_domain_contains_coord(this->l_domain, l_coord));

    return this->coord_mapping.at_l(l_coord);
  }

  DimCoord<L> at_r(DimCoord<R> const &r_coord) const {
    ASSERT(minimal_dim_domain_contains_coord(this->r_domain, r_coord));

    return this->coord_mapping.at_r(r_coord);
  }

  bool operator==(MinimalDimDomainMapping<L, R> const &other) const {
    return this->tie() == other.tie();
  }

  bool operator!=(MinimalDimDomainMapping<L, R> const &other) const {
    return this->tie() != other.tie();
  }

public:
  bidict<DimCoord<L>, DimCoord<R>> coord_mapping;
  MinimalDimDomain<L> l_domain;
  MinimalDimDomain<R> r_domain;

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

  friend struct ::std::hash<MinimalDimDomainMapping<L, R>>;
};

template <typename L, typename R>
std::string format_as(MinimalDimDomainMapping<L, R> const &m) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return fmt::format(
    "<MinimalDimDomainMapping l_domain={} r_domain={} coord_mapping={}>",
    m.l_domain,
    m.r_domain,
    m.coord_mapping);
}

template <typename L, typename R>
std::ostream &operator<<(std::ostream &s, MinimalDimDomainMapping<L, R> const &m) {
  CHECK_FMTABLE(L);
  CHECK_FMTABLE(R);

  return (s << fmt::to_string(m));
}

template <typename L, typename R>
MinimalDimDomainMapping<L, R> minimal_mapping_from_dim_domain_mapping(
  DimDomainMapping<L, R> const &m) {
  
  std::unordered_set<L> l_nontrivial_dims = 
    get_nontrivial_domain_dims(m.l_domain);

  std::unordered_set<R> r_nontrivial_dims = 
    get_nontrivial_domain_dims(m.r_domain);

  return MinimalDimDomainMapping{
    /*coord_mapping=*/
      transform_keys(
        transform_values(
          m.coord_mapping,
          [&](DimCoord<R> const &r_coord) {
            return restrict_coord_to_dims(r_coord, r_nontrivial_dims);
          }),
        [&](DimCoord<L> const &l_coord) {
          return restrict_coord_to_dims(l_coord, l_nontrivial_dims);
        }),
    /*l_domain=*/minimal_dim_domain_from_dim_domain(m.l_domain),
    /*r_domain=*/minimal_dim_domain_from_dim_domain(m.r_domain),
  };
}

template <typename L, typename R>
DimDomainMapping<L, R> dim_domain_mapping_from_minimal_dim_domain(
  MinimalDimDomainMapping<L, R> const &m,
  std::unordered_set<L> const &l_trivial_dims,
  std::unordered_set<R> const &r_trivial_dims) {
  

  DimDomain<L> l_domain = dim_domain_from_minimal_dim_domain(m.l_domain, l_trivial_dims);
  DimDomain<R> r_domain = dim_domain_from_minimal_dim_domain(m.r_domain, r_trivial_dims);

  std::unordered_set<L> all_l_dims = get_domain_dims(l_domain);
  std::unordered_set<R> all_r_dims = get_domain_dims(r_domain);

  return DimDomainMapping{
    /*coord_mapping=*/
      transform_keys(
        transform_values(
          m.coord_mapping,
          [&](DimCoord<R> const &r_coord) {
            return lift_dim_coord(r_coord, all_r_dims);
          }),
        [&](DimCoord<L> const &l_coord) {
          return lift_dim_coord(l_coord, all_l_dims);
        }),
    /*l_domain=*/l_domain,
    /*r_domain=*/r_domain,
  };
}

template <typename L, typename R>
MinimalDimDomainMapping<L, R> empty_minimal_dim_domain_mapping() {
  return MinimalDimDomainMapping{
    /*coord_mapping=*/{},
    /*l_domain=*/empty_minimal_dim_domain<L>(),
    /*r_domain=*/empty_minimal_dim_domain<R>(),
  };
}

template <typename L, typename R>
MinimalDimDomainMapping<L, R> minimal_dim_domain_mapping_identity_map(
    MinimalDimDomain<L> const &l_domain, 
    MinimalDimDomain<R> const &r_domain, 
    DimOrdering<L> const &l_dim_ordering, 
    DimOrdering<R> const &r_dim_ordering) {
  DimProjection<L, R> projection =
    dim_projection_identity_map(
      lift_minimal_dim_domain(l_domain), 
      lift_minimal_dim_domain(r_domain), 
      l_dim_ordering, 
      r_dim_ordering);

  return minimal_dim_domain_mapping_from_projection(
    /*projection=*/projection,
    /*l_domain=*/l_domain,
    /*r_domain=*/r_domain,
    /*l_dim_ordering=*/l_dim_ordering,
    /*r_dim_ordering=*/r_dim_ordering);
}

template <typename L, typename R>
MinimalDimDomainMapping<R, L> invert_minimal_dim_domain_mapping(
  MinimalDimDomainMapping<L, R> const &minimal_dim_domain_mapping) {
  
  return MinimalDimDomainMapping{
    /*coord_mapping=*/minimal_dim_domain_mapping.coord_mapping.reversed(),
    /*l_domain=*/minimal_dim_domain_mapping.r_domain,
    /*r_domain=*/minimal_dim_domain_mapping.l_domain,
  };
}

template <typename T1, typename T2, typename T3>
MinimalDimDomainMapping<T1, T3> compose_minimal_dim_domain_mappings(
  MinimalDimDomainMapping<T1, T2> const &lhs,
  MinimalDimDomainMapping<T2, T3> const &rhs) {
 
  ASSERT(lhs.r_domain == rhs.l_domain);
  
  return MinimalDimDomainMapping{
    /*coord_mapping=*/exhaustive_relational_join(
      lhs.coord_mapping,
      rhs.coord_mapping),
    /*l_domain=*/lhs.l_domain,
    /*r_domain=*/rhs.r_domain,
  };
}


template <typename L, typename R>
MinimalDimDomainMapping<L, R> minimal_dim_domain_mapping_from_projection(
  DimProjection<L, R> const &projection,
  MinimalDimDomain<L> const &l_domain,
  MinimalDimDomain<R> const &r_domain,
  DimOrdering<L> const &l_dim_ordering,
  DimOrdering<R> const &r_dim_ordering) {
  
  return MinimalDimDomainMapping{
    /*coord_mapping=*/generate_bidict(
      get_coords_in_minimal_dim_domain(l_domain),
      [&](DimCoord<L> const &l_coord) {
        return compute_dim_projection(
          /*projection=*/projection,
          /*input_coord=*/l_coord,
          /*input_domain=*/lift_minimal_dim_domain(l_domain),
          /*output_domain=*/lift_minimal_dim_domain(r_domain),
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
struct hash<::FlexFlow::MinimalDimDomainMapping<L, R>> {
  size_t operator()(::FlexFlow::MinimalDimDomainMapping<L, R> const &minimal_dim_domain_mapping) const {
    return get_std_hash(minimal_dim_domain_mapping.tie()); 
  }
};

} // namespace std

#endif
