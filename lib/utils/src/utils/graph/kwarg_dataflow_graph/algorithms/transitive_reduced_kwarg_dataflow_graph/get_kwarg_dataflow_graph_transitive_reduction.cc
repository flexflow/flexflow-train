#include "utils/graph/kwarg_dataflow_graph/algorithms/transitive_reduced_kwarg_dataflow_graph/get_kwarg_dataflow_graph_transitive_reduction.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template TransitiveReducedKwargDataflowGraphView<SlotName>
    get_kwarg_dataflow_graph_transitive_reduction(
        KwargDataflowGraphView<SlotName> const &);

} // namespace FlexFlow
