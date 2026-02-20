#include "utils/graph/kwarg_dataflow_graph/algorithms/find_isomorphism_between_kwarg_dataflow_graphs.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template std::optional<bidict<Node, Node>>
    find_isomorphism_between_kwarg_dataflow_graphs(
        KwargDataflowGraphView<SlotName> const &,
        KwargDataflowGraphView<SlotName> const &);

} // namespace FlexFlow
