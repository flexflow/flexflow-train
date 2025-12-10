#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_OPEN_KWARG_DATAFLOW_EDGES_FOR_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_OPEN_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_INCOMING_OPEN_KWARG_DATAFLOW_EDGES_FOR_NODE_H

#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_graph_view.h"
#include "utils/containers/unordered_map_from_pairs.h"
#include "utils/graph/open_kwarg_dataflow_graph/open_kwarg_dataflow_edge.h"

namespace FlexFlow {

template <typename GraphInputName, typename SlotName>
std::unordered_map<SlotName, OpenKwargDataflowEdge<GraphInputName, SlotName>>
  get_incoming_open_kwarg_dataflow_edges_for_node(
    OpenKwargDataflowGraphView<GraphInputName, SlotName> const &g,
    Node const &n)
{
  OpenKwargDataflowEdgeQuery<GraphInputName, SlotName> query = 
    OpenKwargDataflowEdgeQuery<GraphInputName, SlotName>{
      /*input_edge_query=*/
        KwargDataflowInputEdgeQuery<GraphInputName, SlotName>{
          /*srcs=*/query_set<GraphInputName>::matchall(),
          /*dst_nodes=*/query_set<Node>{n},
          /*dst_slots=*/query_set<SlotName>::matchall(),
        },
      /*standard_edge_query=*/
        KwargDataflowEdgeQuery<SlotName>{
          /*src_nodes=*/query_set<Node>::matchall(),
          /*src_slots=*/query_set<SlotName>::matchall(),
          /*dst_nodes=*/query_set<Node>{n},
          /*dst_slots=*/query_set<SlotName>::matchall(),
        },
    };

  return unordered_map_from_pairs(
    transform(g.query_edges(query),
              [](OpenKwargDataflowEdge<GraphInputName, SlotName> const &e) {
                return std::pair{
                  get_dst_of_open_kwarg_dataflow_edge(e).slot_name,
                  e,
                };
              }));
}

} // namespace FlexFlow

#endif
