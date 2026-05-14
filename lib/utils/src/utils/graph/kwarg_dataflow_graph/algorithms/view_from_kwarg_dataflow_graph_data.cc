#include "utils/graph/kwarg_dataflow_graph/algorithms/view_from_kwarg_dataflow_graph_data.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template KwargDataflowGraphView<SlotName> view_from_kwarg_dataflow_graph_data(
    KwargDataflowGraphData<SlotName> const &);

} // namespace FlexFlow
