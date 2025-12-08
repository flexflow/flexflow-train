#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_OPEN_KWARG_DATAFLOW_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_OPEN_KWARG_DATAFLOW_EDGES_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge_query.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
  get_all_open_kwarg_dataflow_edges(OpenKwargDataflowGraphView<GraphInputName, SlotName> const &view) 
{
  return view.query_edges(open_kwarg_dataflow_edge_query_all<GraphInputName, SlotName>());
}

} // namespace FlexFlow

#endif
