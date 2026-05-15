#include "utils/graph/kwarg_dataflow_graph/algorithms/kwarg_dataflow_graph_data.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template void require_kwarg_dataflow_graph_data_is_valid(
    KwargDataflowGraphData<SlotName> const &);

} // namespace FlexFlow
