#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_subgraph.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template OpenKwargDataflowSubgraphResult<GraphInputName, SlotName>
    get_open_kwarg_dataflow_graph_subgraph(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
        std::unordered_set<Node> const &,
        std::function<GraphInputName()> const &);

template bidict<OpenKwargDataflowValue<GraphInputName, SlotName>,
                KwargDataflowGraphInput<GraphInputName>>
    get_full_kwarg_dataflow_graph_values_to_subgraph_inputs(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
        std::unordered_set<Node> const &,
        std::function<GraphInputName()> const &);

template OpenKwargDataflowGraphData<GraphInputName, SlotName>
    get_open_kwarg_dataflow_subgraph_data(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
        std::unordered_set<Node> const &,
        bidict<OpenKwargDataflowValue<GraphInputName, SlotName>,
               KwargDataflowGraphInput<GraphInputName>> const &);

} // namespace FlexFlow
