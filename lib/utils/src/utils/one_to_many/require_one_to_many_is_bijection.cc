#include "utils/one_to_many/require_one_to_many_is_bijection.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template
  bidict<L, R> require_one_to_many_is_bijection(OneToMany<L, R> const &);

} // namespace FlexFlow
