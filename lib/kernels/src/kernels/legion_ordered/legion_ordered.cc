#include "kernels/legion_ordered/legion_ordered.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using T = value_type<0>;

template struct LegionOrdered<T>;

} // namespace FlexFlow
