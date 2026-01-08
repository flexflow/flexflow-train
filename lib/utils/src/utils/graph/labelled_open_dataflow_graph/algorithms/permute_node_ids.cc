#include "utils/graph/labelled_open_dataflow_graph/algorithms/permute_node_ids.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using ValueLabel = value_type<1>;

template LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> permute_node_ids(
    LabelledOpenDataflowGraphView<NodeLabel, ValueLabel> const &,
    bidict<NewNode, Node> const &);

} // namespace FlexFlow
