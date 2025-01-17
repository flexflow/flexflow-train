#include "utils/many_to_one/invert_many_to_one.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  OneToMany<R, L> invert_many_to_one(ManyToOne<L, R> const &);

} // namespace FlexFlow
