#include "utils/one_to_many/invert_one_to_many.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  ManyToOne<R, L> invert_one_to_many(OneToMany<L, R> const &);

} // namespace FlexFlow
