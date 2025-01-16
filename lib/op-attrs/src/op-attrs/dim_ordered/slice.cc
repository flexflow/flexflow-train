#include "op-attrs/dim_ordered/slice.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template FFOrdered<T>
    ff_dim_t_nonoverloaded_slice(FFOrdered<T> const &d,
                                 std::optional<ff_dim_t> const &start,
                                 std::optional<ff_dim_t> const &end);

template FFOrdered<T> relative_ff_dim_t_nonoverloaded_slice(
    FFOrdered<T> const &d,
    std::optional<relative_ff_dim_t> const &start,
    std::optional<relative_ff_dim_t> const &end);

template FFOrdered<T> slice(FFOrdered<T> const &d,
                            std::optional<ff_dim_t> const &start,
                            std::optional<ff_dim_t> const &end);

template FFOrdered<T> slice(FFOrdered<T> const &d,
                            std::optional<relative_ff_dim_t> const &start,
                            std::optional<relative_ff_dim_t> const &end);

} // namespace FlexFlow
