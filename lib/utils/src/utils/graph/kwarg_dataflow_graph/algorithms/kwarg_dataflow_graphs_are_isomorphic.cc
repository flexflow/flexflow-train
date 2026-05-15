#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graphs_are_isomorphic.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template bool kwarg_dataflow_graphs_are_isomorphic(
    KwargDataflowGraphView<SlotName> const &,
    KwargDataflowGraphView<SlotName> const &);

} // namespace FlexFlow
