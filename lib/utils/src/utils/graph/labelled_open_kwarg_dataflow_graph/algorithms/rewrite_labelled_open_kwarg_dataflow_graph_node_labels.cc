#include "utils/graph/labelled_open_kwarg_dataflow_graph/algorithms/rewrite_labelled_open_kwarg_dataflow_graph_node_labels.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using NewNodeLabel = value_type<1>;
using ValueLabel = value_type<2>;
using GraphInputName = ordered_value_type<3>;
using SlotName = ordered_value_type<4>;
using F = std::function<NewNodeLabel(Node const &, NodeLabel const &)>;

template LabelledOpenKwargDataflowGraphView<NewNodeLabel,
                                            ValueLabel,
                                            GraphInputName,
                                            SlotName>
    rewrite_labelled_open_kwarg_dataflow_graph_node_labels(
        LabelledOpenKwargDataflowGraphView<NodeLabel,
                                           ValueLabel,
                                           GraphInputName,
                                           SlotName> const &,
        F);

} // namespace FlexFlow
