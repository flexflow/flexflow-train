#include "utils/graph/labelled_dataflow_graph/algorithms/rewrite_node_labels.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;
using NewNodeLabel = value_type<2>;
using F = std::function<NewNodeLabel(Node const &, NodeLabel const &)>;

template LabelledDataflowGraphView<NewNodeLabel, ValueLabel>
    rewrite_node_labels(
        LabelledDataflowGraphView<NodeLabel, ValueLabel> const &, F);

} // namespace FlexFlow
