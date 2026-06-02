#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_graph_subgraph.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template KwargDataflowGraphView<SlotName>
    get_kwarg_dataflow_graph_subgraph(KwargDataflowGraphView<SlotName> const &,
                                      std::unordered_set<Node> const &);

} // namespace FlexFlow
