#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_KWARG_DATAFLOW_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_ALL_KWARG_DATAFLOW_EDGES_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_set<KwargDataflowEdge<SlotName>> get_all_kwarg_dataflow_edges(
  KwargDataflowGraphView<SlotName> const &g) 
{
  return g.query_edges(kwarg_dataflow_edge_query_all<SlotName>());
}

} // namespace FlexFlow

#endif
