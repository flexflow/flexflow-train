#include "utils/orthotope/minimal_dim_domain_mapping.h"
#include "utils/archetypes/ordered_value_type.h"

using ::FlexFlow::ordered_value_type;
using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

namespace FlexFlow {

template struct MinimalDimDomainMapping<L, R>;

template std::string format_as(MinimalDimDomainMapping<L, R> const &);

template std::ostream &operator<<(std::ostream &,
                                  MinimalDimDomainMapping<L, R> const &);

template MinimalDimDomainMapping<L, R>
    minimal_mapping_from_dim_domain_mapping(DimDomainMapping<L, R> const &);

template DimDomainMapping<L, R> dim_domain_mapping_from_minimal_dim_domain(
    MinimalDimDomainMapping<L, R> const &,
    std::unordered_set<L> const &,
    std::unordered_set<R> const &);

template MinimalDimDomainMapping<L, R>
    minimal_dim_domain_mapping_identity_map(MinimalDimDomain<L> const &,
                                            MinimalDimDomain<R> const &,
                                            DimOrdering<L> const &,
                                            DimOrdering<R> const &);

template MinimalDimDomainMapping<L, R> empty_minimal_dim_domain_mapping();

template MinimalDimDomainMapping<R, L>
    invert_minimal_dim_domain_mapping(MinimalDimDomainMapping<L, R> const &);

template MinimalDimDomainMapping<L, R>
    minimal_dim_domain_mapping_from_projection(DimProjection<L, R> const &,
                                               MinimalDimDomain<L> const &,
                                               MinimalDimDomain<R> const &,
                                               DimOrdering<L> const &,
                                               DimOrdering<R> const &);

using T1 = ordered_value_type<2>;
using T2 = ordered_value_type<3>;
using T3 = ordered_value_type<4>;

template MinimalDimDomainMapping<T1, T3> compose_minimal_dim_domain_mappings(
    MinimalDimDomainMapping<T1, T2> const &,
    MinimalDimDomainMapping<T2, T3> const &);

template DimDomainMapping<T1, T3> compose_dim_domain_mappings_through_minimal(
    DimDomainMapping<T1, T2> const &, DimDomainMapping<T2, T3> const &);

} // namespace FlexFlow

namespace std {

template struct hash<::FlexFlow::MinimalDimDomainMapping<L, R>>;

}
