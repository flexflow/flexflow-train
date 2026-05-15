#include "utils/graph/instances/unordered_set_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template struct UnorderedSetKwargDataflowGraph<SlotName>;

} // namespace FlexFlow
