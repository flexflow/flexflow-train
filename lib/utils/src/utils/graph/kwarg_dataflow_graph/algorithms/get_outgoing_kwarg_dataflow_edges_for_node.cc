#include "utils/graph/kwarg_dataflow_graph/algorithms/get_outgoing_kwarg_dataflow_edges_for_node.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template OneToMany<SlotName, KwargDataflowEdge<SlotName>>
    get_outgoing_kwarg_dataflow_edges_for_node(
        KwargDataflowGraphView<SlotName> const &, Node const &);

} // namespace FlexFlow
