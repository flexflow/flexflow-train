#include "utils/unordered_bidict/algorithms/unordered_bidict_exhaustive_relational_join.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;

template unordered_bidict<T1, T3> unordered_bidict_exhaustive_relational_join(
    unordered_bidict<T1, T2> const &, unordered_bidict<T2, T3> const &);

} // namespace FlexFlow
