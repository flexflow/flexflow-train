#include "utils/one_to_many/one_to_many_from_bidict.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template OneToMany<L, R> one_to_many_from_bidict(bidict<L, R> const &);

} // namespace FlexFlow
