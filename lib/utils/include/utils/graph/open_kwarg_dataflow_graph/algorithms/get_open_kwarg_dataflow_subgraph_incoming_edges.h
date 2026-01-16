#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_KWARG_DATAFLOW_SUBGRAPH_INCOMING_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OPEN_KWARG_DATAFLOW_SUBGRAPH_INCOMING_EDGES_H

#include "utils/containers/set_minus.h"
#include "utils/graph/node/algorithms.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::unordered_set<OpenKwargDataflowEdge<GraphInputName, SlotName>>
    get_open_kwarg_dataflow_subgraph_incoming_edges(
        OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
        std::unordered_set<Node> const &subgraph) {
  std::unordered_set<Node> all_nodes = get_nodes(g);
  query_set<Node> src_query = query_set<Node>{set_minus(all_nodes, subgraph)};

  OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> query =
      OpenKwargDataflowEdgeQuery<GraphInputName, SlotName>{
          /*input_edge_query=*/KwargDataflowInputEdgeQuery<GraphInputName,
                                                           SlotName>{
              /*srcs=*/query_set<GraphInputName>::matchall(),
              /*dst_nodes=*/query_set<Node>{subgraph},
              /*dst_slots=*/query_set<SlotName>::matchall(),
          },
          /*standard_edge_query=*/
          KwargDataflowEdgeQuery<SlotName>{
              /*src_nodes=*/src_query,
              /*src_slots=*/query_set<SlotName>::matchall(),
              /*dst_nodes=*/query_set<Node>{subgraph},
              /*dst_slots=*/query_set<SlotName>::matchall(),
          },
      };

  return g.query_edges(query);
}

} // namespace FlexFlow

#endif
