#include "op-attrs/ff_ordered/ff_ordered_of.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template
  FFOrdered<T> ff_ordered_of(std::vector<T> const &);

template
  FFOrdered<T> ff_ordered_of(std::unordered_set<T> const &);


} // namespace FlexFlow
