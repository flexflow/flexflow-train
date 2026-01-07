#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_SUBGRAPH_OUTGOING_EDGES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_SUBGRAPH_OUTGOING_EDGES_H

#include "utils/containers/set_minus.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_set<KwargDataflowEdge<SlotName>>
    get_kwarg_dataflow_subgraph_outgoing_edges(
        KwargDataflowGraphView<SlotName> const &g,
        std::unordered_set<Node> const &subgraph) {
  std::unordered_set<Node> all_nodes = get_nodes(g);
  query_set<Node> dst_query = query_set<Node>{set_minus(all_nodes, subgraph)};

  KwargDataflowEdgeQuery<SlotName> query = KwargDataflowEdgeQuery<SlotName>{
      /*src_nodes=*/query_set<Node>{subgraph},
      /*src_slots=*/query_set<SlotName>::matchall(),
      /*dst_nodes=*/dst_query,
      /*dst_slots=*/query_set<SlotName>::matchall(),
  };

  return g.query_edges(query);
}

} // namespace FlexFlow

#endif
