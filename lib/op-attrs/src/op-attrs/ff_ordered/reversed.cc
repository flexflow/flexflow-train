#include "op-attrs/ff_ordered/reversed.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template 
  FFOrdered<T> reversed(FFOrdered<T> const &);

} // namespace FlexFlow
