#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_all_open_kwarg_dataflow_edges.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
    get_all_open_kwarg_dataflow_edges(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &);

} // namespace FlexFlow
