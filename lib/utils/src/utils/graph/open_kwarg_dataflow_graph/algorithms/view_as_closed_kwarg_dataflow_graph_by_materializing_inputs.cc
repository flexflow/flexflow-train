#include "utils/graph/open_kwarg_dataflow_graph/algorithms/view_as_closed_kwarg_dataflow_graph_by_materializing_inputs.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template std::pair<KwargDataflowGraphView<std::optional<SlotName>>,
                   bidict<KwargDataflowGraphInput<GraphInputName>, Node>>
    view_as_closed_kwarg_dataflow_graph_by_materializing_inputs(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &);

} // namespace FlexFlow
