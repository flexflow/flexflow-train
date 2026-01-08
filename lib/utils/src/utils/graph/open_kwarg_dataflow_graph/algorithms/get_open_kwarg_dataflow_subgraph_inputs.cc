#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_subgraph_inputs.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

std::unordered_set<OpenKwargDataflowValue<GraphInputName, SlotName>>
    get_open_kwarg_dataflow_subgraph_inputs(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
        std::unordered_set<Node> const &);

} // namespace FlexFlow
