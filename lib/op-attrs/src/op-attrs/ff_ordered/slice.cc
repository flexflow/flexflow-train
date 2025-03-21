#include "op-attrs/ff_ordered/slice.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template FFOrdered<T> ff_dim_t_nonoverloaded_slice(
    FFOrdered<T> const &, ff_dim_t const &, std::optional<ff_dim_t> const &);

template FFOrdered<T> relative_ff_dim_t_nonoverloaded_slice(
    FFOrdered<T> const &,
    relative_ff_dim_t const &,
    std::optional<relative_ff_dim_t> const &);

template FFOrdered<T> slice(FFOrdered<T> const &,
                            ff_dim_t const &,
                            std::optional<ff_dim_t> const &);

template FFOrdered<T> slice(FFOrdered<T> const &,
                            relative_ff_dim_t const &,
                            std::optional<relative_ff_dim_t> const &);

} // namespace FlexFlow
