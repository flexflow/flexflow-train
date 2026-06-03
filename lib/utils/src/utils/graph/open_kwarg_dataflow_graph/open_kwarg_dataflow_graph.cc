#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template struct OpenKwargDataflowGraph<GraphInputName, SlotName>;

} // namespace FlexFlow
