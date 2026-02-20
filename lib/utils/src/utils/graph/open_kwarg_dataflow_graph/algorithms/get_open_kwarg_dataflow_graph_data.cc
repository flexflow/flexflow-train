#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_open_kwarg_dataflow_graph_data.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template OpenKwargDataflowGraphData<GraphInputName, SlotName>
    get_open_kwarg_dataflow_graph_data(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &);

} // namespace FlexFlow
