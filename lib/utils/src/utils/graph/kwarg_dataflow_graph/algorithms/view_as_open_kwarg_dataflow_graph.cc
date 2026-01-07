#include "utils/graph/kwarg_dataflow_graph/algorithms/view_as_open_kwarg_dataflow_graph.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template OpenKwargDataflowGraphView<GraphInputName, SlotName>
    view_as_open_kwarg_dataflow_graph(KwargDataflowGraphView<SlotName> const &);

} // namespace FlexFlow
