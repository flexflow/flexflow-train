#include "op-attrs/ff_ordered/ff_ordered.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template struct FFOrdered<T>;

template std::string format_as(FFOrdered<T> const &);

template std::ostream &operator<<(std::ostream &, FFOrdered<T> const &);

} // namespace FlexFlow
