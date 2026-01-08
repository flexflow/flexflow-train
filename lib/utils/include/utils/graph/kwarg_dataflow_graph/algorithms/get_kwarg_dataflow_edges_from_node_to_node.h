#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_EDGES_FROM_NODE_TO_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_EDGES_FROM_NODE_TO_NODE_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_set<KwargDataflowEdge<SlotName>>
    get_kwarg_dataflow_edges_from_node_to_node(
        KwargDataflowGraphView<SlotName> const &g,
        Node const &src,
        Node const &dst) {
  KwargDataflowEdgeQuery<SlotName> query = KwargDataflowEdgeQuery<SlotName>{
      /*src_nodes=*/query_set<Node>{src},
      /*src_slots=*/query_set<SlotName>::matchall(),
      /*dst_nodes=*/query_set<Node>{dst},
      /*dst_slots=*/query_set<SlotName>::matchall(),
  };

  return g.query_edges(query);
}

} // namespace FlexFlow

#endif
