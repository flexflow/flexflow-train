#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_VALUE_USES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_VALUE_USES_H

#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph_view.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_set<KwargDataflowInput<SlotName>>
    get_kwarg_dataflow_value_uses(KwargDataflowGraphView<SlotName> const &g,
                                  KwargDataflowOutput<SlotName> const &v) {

  KwargDataflowEdgeQuery<SlotName> query = KwargDataflowEdgeQuery<SlotName>{
      /*src_nodes=*/query_set<Node>::match_single_value(v.node),
      /*src_slots=*/query_set<SlotName>::match_single_value(v.slot_name),
      /*dst_nodes=*/query_set<Node>::matchall(),
      /*dst_slots=*/query_set<SlotName>::matchall(),
  };

  std::unordered_set<KwargDataflowEdge<SlotName>> edges = g.query_edges(query);

  return transform(edges,
                   [&](KwargDataflowEdge<SlotName> const &e) { return e.dst; });
}

} // namespace FlexFlow

#endif
