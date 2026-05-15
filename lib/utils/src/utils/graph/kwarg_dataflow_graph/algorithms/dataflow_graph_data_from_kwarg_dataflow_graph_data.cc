#include "utils/graph/kwarg_dataflow_graph/algorithms/dataflow_graph_data_from_kwarg_dataflow_graph_data.h"
#include "utils/archetypes/ordered_value_type.h"

namespace FlexFlow {

using SlotName = ordered_value_type<0>;

template DataflowGraphData dataflow_graph_data_from_kwarg_dataflow_graph_data(
    KwargDataflowGraphData<SlotName> const &,
    std::function<
        std::vector<SlotName>(std::unordered_set<SlotName> const &)> const &);

} // namespace FlexFlow
