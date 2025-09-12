#include "utils/orthotope/minimal_dim_domain.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template MinimalDimDomain<T> empty_minimal_dim_domain();

template nonnegative_int minimal_dim_domain_num_dims(MinimalDimDomain<T> const &);

template
  DimDomain<T> lift_minimal_dim_domain(MinimalDimDomain<T> const &);

template
  MinimalDimDomain<T> require_dim_domain_is_minimal(DimDomain<T> const &);

template
  MinimalDimDomain<T> minimal_dim_domain_from_dim_domain(DimDomain<T> const &);

template
  DimDomain<T> dim_domain_from_minimal_dim_domain(MinimalDimDomain<T> const &,
                                                  std::unordered_set<T> const &);

template
  std::unordered_set<T> get_minimal_domain_dims(MinimalDimDomain<T> const &);

template
  MinimalDimDomain<T> restrict_minimal_domain_to_dims(MinimalDimDomain<T> const &,
                                                      std::unordered_set<T> const &);

template
  MinimalOrthotope minimal_orthotope_from_minimal_dim_domain(
    MinimalDimDomain<T> const &,
    DimOrdering<T> const &);

template
  MinimalDimDomain<T> minimal_dim_domain_from_minimal_orthotope(
    MinimalOrthotope const &,
    std::unordered_set<T> const &,
    DimOrdering<T> const &);

} // namespace FlexFlow
