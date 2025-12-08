#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_KWARG_DATAFLOW_INPUT_EDGE_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_KWARG_DATAFLOW_INPUT_EDGE_QUERY_H

#include "utils/graph/open_kwarg_dataflow_graph/kwarg_dataflow_input_edge.dtg.h"
#include "utils/graph/open_kwarg_dataflow_graph/kwarg_dataflow_input_edge_query.dtg.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
KwargDataflowInputEdgeQuery<GraphInputName, SlotName>
  kwarg_dataflow_input_edge_query_all() 
{
  return KwargDataflowInputEdgeQuery<GraphInputName, SlotName>{
    /*srcs=*/query_set<GraphInputName>::matchall(),
    /*dst_nodes=*/query_set<Node>::matchall(),
    /*dst_idxs=*/query_set<SlotName>::matchall(),
  };
}

template <typename GraphInputName, typename SlotName>
bool kwarg_dataflow_input_edge_query_includes(
  KwargDataflowInputEdgeQuery<GraphInputName, SlotName> const &query,
  KwargDataflowInputEdge<GraphInputName, SlotName> const &edge)
{
  return includes(query.srcs, edge.src.name)
    && includes(query.dst_nodes, edge.dst.node)
    && includes(query.dst_idxs, edge.dst.idx);
}

} // namespace FlexFlow

#endif
