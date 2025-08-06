#include "utils/orthotope/dim_domain_mapping.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template struct DimDomainMapping<L, R>;

template
  DimDomainMapping<R, L> invert_dim_domain_mapping(
    DimDomainMapping<L, R> const &);

using T1 = value_type<2>;
using T2 = value_type<3>;
using T3 = value_type<4>;

template
  DimDomainMapping<T1, T3> compose_dim_domain_mappings(
    DimDomainMapping<T1, T2> const &,
    DimDomainMapping<T2, T3> const &);

} // namespace FlexFlow
