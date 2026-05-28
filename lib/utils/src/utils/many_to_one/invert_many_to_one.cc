#include "utils/many_to_one/invert_many_to_one.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using L = ordered_value_type<0>;
using R = ordered_value_type<1>;

template OneToMany<R, L> invert_many_to_one(ManyToOne<L, R> const &);

} // namespace FlexFlow
