#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/rewrite_labelled_kwarg_dataflow_graph_node_labels.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using SlotName = ordered_value_type<2>;
using NewNodeLabel = value_type<3>;
using F = std::function<NewNodeLabel(NodeLabel const &)>;

LabelledKwargDataflowGraphView<NewNodeLabel, ValueLabel, SlotName>
    rewrite_labelled_kwarg_dataflow_graph_node_labels(
        LabelledKwargDataflowGraphView<NodeLabel, ValueLabel, SlotName> const &,
        F);

} // namespace FlexFlow
