#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_graph_data.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template KwargDataflowGraphData<SlotName>
    get_kwarg_dataflow_graph_data(KwargDataflowGraphView<SlotName> const &);

} // namespace FlexFlow
