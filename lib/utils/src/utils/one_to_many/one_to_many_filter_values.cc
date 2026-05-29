#include "utils/one_to_many/one_to_many_filter_values.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

using F = std::function<bool(R const &)>;

template OneToMany<L, R> one_to_many_filter_values(OneToMany<L, R> const &, F &&);

} // namespace FlexFlow
