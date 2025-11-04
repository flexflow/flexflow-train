#include "op-attrs/ff_ordered/concat.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  FFOrdered<T> concat(FFOrdered<T> const &, FFOrdered<T> const &);

template
  FFOrdered<T> concat(std::vector<FFOrdered<T>> const &);

} // namespace FlexFlow
