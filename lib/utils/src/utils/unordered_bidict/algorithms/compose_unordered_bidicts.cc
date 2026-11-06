#include "utils/unordered_bidict/algorithms/compose_unordered_bidicts.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;

template unordered_bidict<T1, T3>
    compose_unordered_bidicts(unordered_bidict<T1, T2> const &,
                              unordered_bidict<T2, T3> const &);

} // namespace FlexFlow
