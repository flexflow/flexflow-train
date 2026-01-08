#include "utils/required_core.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template struct required<int>;
template struct required<T>;

} // namespace FlexFlow
