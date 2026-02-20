#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/labelled_open_kwarg_dataflow_graph_data.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using GraphInputName = ordered_value_type<2>;
using SlotName = ordered_value_type<3>;

template OpenKwargDataflowGraphData<GraphInputName, SlotName>
    labelled_open_kwarg_dataflow_graph_data_without_labels(
        LabelledOpenKwargDataflowGraphData<NodeLabel,
                                           ValueLabel,
                                           GraphInputName,
                                           SlotName> const &);

template void require_labelled_open_kwarg_dataflow_graph_data_is_valid(
    LabelledOpenKwargDataflowGraphData<NodeLabel,
                                       ValueLabel,
                                       GraphInputName,
                                       SlotName> const &);

} // namespace FlexFlow
