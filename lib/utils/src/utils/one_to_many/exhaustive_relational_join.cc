#include "utils/one_to_many/exhaustive_relational_join.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T1 = value_type<0>;
using T2 = value_type<1>;
using T3 = value_type<2>;

template
  OneToMany<T1, T3> exhaustive_relational_join(OneToMany<T1, T2> const &, OneToMany<T2, T3> const &);

} // namespace FlexFlow
