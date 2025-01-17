#include "utils/many_to_one/many_to_one_from_bidict.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using L = value_type<0>;
using R = value_type<1>;

template
  ManyToOne<L, R> many_to_one_from_bidict(bidict<L, R> const &);

} // namespace FlexFlow
