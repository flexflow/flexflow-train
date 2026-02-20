#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_KWARG_DATAFLOW_EDGE_QUERY_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_KWARG_DATAFLOW_EDGE_QUERY_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_edge_query.dtg.h"

namespace FlexFlow {

template <typename SlotName>
KwargDataflowEdgeQuery<SlotName> kwarg_dataflow_edge_query_all() {
  return KwargDataflowEdgeQuery<SlotName>{
      /*src_nodes=*/query_set<Node>::matchall(),
      /*src_slots=*/query_set<SlotName>::matchall(),
      /*dst_nodes=*/query_set<Node>::matchall(),
      /*dst_slots=*/query_set<SlotName>::matchall(),
  };
}

template <typename SlotName>
KwargDataflowEdgeQuery<SlotName> kwarg_dataflow_edge_query_none() {
  return KwargDataflowEdgeQuery<SlotName>{
      /*src_nodes=*/query_set<Node>::match_none(),
      /*src_slots=*/query_set<SlotName>::match_none(),
      /*dst_nodes=*/query_set<Node>::match_none(),
      /*dst_slots=*/query_set<SlotName>::match_none(),
  };
}

template <typename SlotName>
bool kwarg_dataflow_edge_query_includes(
    KwargDataflowEdgeQuery<SlotName> const &query,
    KwargDataflowEdge<SlotName> const &edge) {
  return includes(query.src_nodes, edge.src.node) &&
         includes(query.src_slots, edge.src.slot_name) &&
         includes(query.dst_nodes, edge.dst.node) &&
         includes(query.dst_slots, edge.dst.slot_name);
}

} // namespace FlexFlow

#endif
