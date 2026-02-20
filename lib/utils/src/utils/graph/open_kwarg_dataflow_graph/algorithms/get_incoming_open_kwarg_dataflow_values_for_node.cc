#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_incoming_open_kwarg_dataflow_values_for_node.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;
using GraphInputName = ordered_value_type<1>;

template std::unordered_map<SlotName,
                            OpenKwargDataflowValue<GraphInputName, SlotName>>
    get_incoming_open_kwarg_dataflow_values_for_node(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &,
        Node const &);

} // namespace FlexFlow
