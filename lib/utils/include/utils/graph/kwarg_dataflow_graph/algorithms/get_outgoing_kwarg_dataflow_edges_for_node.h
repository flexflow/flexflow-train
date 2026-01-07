#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OUTGOING_KWARG_DATAFLOW_EDGES_FOR_NODE_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_OUTGOING_KWARG_DATAFLOW_EDGES_FOR_NODE_H

#include "utils/containers/group_by.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"
#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename SlotName>
OneToMany<SlotName, KwargDataflowEdge<SlotName>>
    get_outgoing_kwarg_dataflow_edges_for_node(
        KwargDataflowGraphView<SlotName> const &g, Node const &n) {
  KwargDataflowEdgeQuery<SlotName> query = KwargDataflowEdgeQuery<SlotName>{
      /*src_nodes=*/query_set<Node>{n},
      /*src_slots=*/query_set<SlotName>::matchall(),
      /*dst_nodes=*/query_set<Node>::matchall(),
      /*dst_slots=*/query_set<SlotName>::matchall(),
  };

  return group_by(
      g.query_edges(query),
      [](KwargDataflowEdge<SlotName> const &e) { return e.src.slot_name; });
}

} // namespace FlexFlow

#endif
