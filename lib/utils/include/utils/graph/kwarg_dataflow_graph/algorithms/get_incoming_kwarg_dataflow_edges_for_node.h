#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_KWARG_DATAFLOW_EDGES_FOR_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_KWARG_DATAFLOW_EDGES_FOR_NODE_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/containers/unordered_map_from_pairs.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_map<SlotName, KwargDataflowEdge<SlotName>> 
  get_incoming_kwarg_dataflow_edges_for_node(
    KwargDataflowGraphView<SlotName> const &g,
    Node const &n)
{
  KwargDataflowEdgeQuery<SlotName> query = 
    KwargDataflowEdgeQuery<SlotName>{
      /*src_nodes=*/query_set<Node>::matchall(),
      /*src_slots=*/query_set<SlotName>::matchall(),
      /*dst_nodes=*/query_set<Node>{n},
      /*dst_slots=*/query_set<SlotName>::matchall(),
    };

  return unordered_map_from_pairs(
    transform(g.query_edges(query),
              [](KwargDataflowEdge<SlotName> const &e) {
                return std::pair{
                  e.dst.slot_name,
                  e,
                };
              }));
}

} // namespace FlexFlow

#endif
