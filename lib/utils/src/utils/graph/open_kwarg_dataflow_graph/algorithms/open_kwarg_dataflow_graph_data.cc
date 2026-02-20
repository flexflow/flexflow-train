#include "utils/graph/open_kwarg_dataflow_graph/algorithms/open_kwarg_dataflow_graph_data.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using GraphInputName = ordered_value_type<0>;
using SlotName = ordered_value_type<1>;

template void require_open_kwarg_dataflow_graph_data_is_valid(
    OpenKwargDataflowGraphData<GraphInputName, SlotName> const &);

} // namespace FlexFlow
