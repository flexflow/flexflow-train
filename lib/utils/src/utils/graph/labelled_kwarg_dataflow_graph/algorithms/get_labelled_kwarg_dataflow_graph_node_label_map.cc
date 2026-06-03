#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/get_labelled_kwarg_dataflow_graph_node_label_map.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using OutputLabel = value_type<1>;
using SlotName = ordered_value_type<2>;

template std::unordered_map<Node, NodeLabel>
    get_labelled_kwarg_dataflow_graph_node_label_map(
        LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName> const
            &);

} // namespace FlexFlow
