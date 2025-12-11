#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_OPEN_KWARG_DATAFLOW_VALUES_FOR_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_OPEN_KWARG_DATAFLOW_VALUES_FOR_NODE_H

#include "utils/graph/open_kwarg_dataflow_graph/algorithms/get_incoming_open_kwarg_dataflow_edges_for_node.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"
#include "utils/containers/map_values.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::unordered_map<SlotName, OpenKwargDataflowValue<GraphInputName, SlotName>>
  get_incoming_open_kwarg_dataflow_values_for_node(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
    Node const &n)
{
  return map_values(
    get_incoming_open_kwarg_dataflow_edges_for_node(g, n),
    [](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) 
      -> OpenKwargDataflowValue<GraphInputName, SlotName>
    {
      return get_src_of_open_kwarg_dataflow_edge(e);
    });
}

} // namespace FlexFlow

#endif
