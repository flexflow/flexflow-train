#include "utils/stack_map.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using K = ordered_value_type<0>;

template struct stack_map<K, int, 5>;

} // namespace FlexFlow
