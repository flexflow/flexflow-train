#include "utils/deduplicated_priority_queue.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using T = ordered_value_type<0>;

template class DeduplicatedPriorityQueue<T>;

} // namespace FlexFlow
