#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/get_labelled_open_kwarg_dataflow_graph_data.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using GraphInputName = ordered_value_type<2>;
using SlotName = ordered_value_type<3>;

template LabelledOpenKwargDataflowGraphData<NodeLabel,
                                            ValueLabel,
                                            GraphInputName,
                                            SlotName>
    get_labelled_open_kwarg_dataflow_graph_data(
        LabelledOpenKwargDataflowGraphView<NodeLabel,
                                           ValueLabel,
                                           GraphInputName,
                                           SlotName> const &);

} // namespace FlexFlow
