#include "utils/orthotope/dim_domain_mapping.h"
#include "utils/archetypes/value_type.h"

using ::FlexFlow::value_type;
using L = value_type<0>;
using R = value_type<1>;


namespace FlexFlow {

template struct DimDomainMapping<L, R>;

template
  std::string format_as(DimDomainMapping<L, R> const &);

template
  std::ostream &operator<<(std::ostream &, DimDomainMapping<L, R> const &);

template DimDomainMapping<L, R> dim_domain_mapping_identity_map(
                                      DimDomain<L> const &, 
                                      DimDomain<R> const &, 
                                      DimOrdering<L> const &, 
                                      DimOrdering<R> const &);

template
  DimDomainMapping<L, R> empty_dim_domain_mapping();

template
  DimDomainMapping<R, L> invert_dim_domain_mapping(
    DimDomainMapping<L, R> const &);

template
  DimDomainMapping<L, R> dim_domain_mapping_from_projection(
    DimProjection<L, R> const &,
    DimDomain<L> const &,
    DimDomain<R> const &,
    DimOrdering<L> const &,
    DimOrdering<R> const &);

using T1 = value_type<2>;
using T2 = value_type<3>;
using T3 = value_type<4>;

template
  DimDomainMapping<T1, T3> compose_dim_domain_mappings(
    DimDomainMapping<T1, T2> const &,
    DimDomainMapping<T2, T3> const &);

} // namespace FlexFlow

namespace std {

template struct hash<::FlexFlow::DimDomainMapping<L, R>>;

}
