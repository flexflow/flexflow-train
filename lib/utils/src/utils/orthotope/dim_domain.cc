#include "utils/orthotope/dim_domain.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template
  std::unordered_set<T> get_domain_dims(DimDomain<T> const &);

template
  DimDomain<T> restrict_domain_to_dims(DimDomain<T> const &, std::unordered_set<T> const &);

template
  Orthotope orthotope_from_dim_domain(DimDomain<T> const &);


} // namespace FlexFlow
