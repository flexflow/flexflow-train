#include "utils/many_to_one/exhaustive_relational_join.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;

template 
  ManyToOne<T1, T3> exhaustive_relational_join(ManyToOne<T1, T2> const &, ManyToOne<T2, T3> const &);

} // namespace FlexFlow
