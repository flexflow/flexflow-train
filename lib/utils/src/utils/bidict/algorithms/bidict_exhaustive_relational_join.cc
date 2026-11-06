#include "utils/bidict/algorithms/bidict_exhaustive_relational_join.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T1 = ordered_value_type<0>;
using T2 = ordered_value_type<1>;
using T3 = ordered_value_type<2>;

template bidict<T1, T3>
    bidict_exhaustive_relational_join(bidict<T1, T2> const &,
                                      bidict<T2, T3> const &);

} // namespace FlexFlow
