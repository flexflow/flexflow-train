#include "utils/graph/labelled_kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_view_with_labelling.h"
#include "utils/archetypes/ordered_value_type.h"
#include "utils/archetypes/value_type.h"

namespace FlexFlow {

using NodeLabel = value_type<0>;
using OutputLabel = value_type<1>;
using SlotName = ordered_value_type<2>;

template LabelledKwargDataflowGraphView<NodeLabel, OutputLabel, SlotName>
    kwarg_dataflow_graph_view_with_labelling(
        KwargDataflowGraphView<SlotName> const &,
        std::unordered_map<Node, NodeLabel> const &,
        std::unordered_map<KwargDataflowOutput<SlotName>, OutputLabel> const &);

} // namespace FlexFlow
