#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_KWARG_DATAFLOW_OUTPUTS_FOR_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_KWARG_DATAFLOW_OUTPUTS_FOR_NODE_H

#include "utils/containers/map_values.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_incoming_kwarg_dataflow_edges_for_node.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_map<SlotName, KwargDataflowOutput<SlotName>>
    get_incoming_kwarg_dataflow_outputs_for_node(
        KwargDataflowGraphView<SlotName> const &g, Node const &n) {
  return map_values(get_incoming_kwarg_dataflow_edges_for_node(g, n),
                    [](KwargDataflowEdge<SlotName> const &e) { return e.src; });
}

} // namespace FlexFlow

#endif
