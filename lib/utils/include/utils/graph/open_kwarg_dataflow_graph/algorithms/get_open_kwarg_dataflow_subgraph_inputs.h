#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_KWARG_DATAFLOW_SUBGRAPH_INPUTS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_KWARG_DATAFLOW_SUBGRAPH_INPUTS_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_value.dtg.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::unordered_set<OpenKwargDataflowValue<GraphInputName, SlotName>>
    get_open_kwarg_dataflow_subgraph_inputs(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
        std::unordered_set<Node> const &subgraph_nodes) {

  return transform(
      get_open_kwarg_dataflow_subgraph_incoming_edges(g, subgraph_nodes),
      [](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) {
        return get_src_of_open_kwarg_dataflow_edge(e);
      });
}

} // namespace FlexFlow

#endif
