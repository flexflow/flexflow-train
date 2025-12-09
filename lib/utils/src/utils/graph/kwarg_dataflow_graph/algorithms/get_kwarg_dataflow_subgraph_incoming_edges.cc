#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_subgraph_incoming_edges.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template
  std::unordered_set<KwargDataflowEdge<SlotName>> get_kwarg_dataflow_subgraph_incoming_edges(
    KwargDataflowGraphView<SlotName> const &,
    std::unordered_set<Node> const &);


} // namespace FlexFlow
