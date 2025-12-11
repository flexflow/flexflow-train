#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_edges_from_node_to_node.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template
  std::unordered_set<KwargDataflowEdge<SlotName>>
    get_kwarg_dataflow_edges_from_node_to_node( 
      KwargDataflowGraphView<SlotName> const &,
      Node const &,
      Node const &);

} // namespace FlexFlow
