#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_GRAPH_SUBGRAPH_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_GET_KWARG_DATAFLOW_GRAPH_SUBGRAPH_H

#include "utils/containers/set_intersection.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_graph_data.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/view_from_kwarg_dataflow_graph_data.h"
#include "utils/graph/kwarg_dataflow_graph/kwarg_dataflow_graph.h"

namespace FlexFlow {

template <typename SlotName>
KwargDataflowGraphView<SlotName> get_kwarg_dataflow_graph_subgraph(
    KwargDataflowGraphView<SlotName> const &g,
    std::unordered_set<Node> const &subgraph_nodes) {
  KwargDataflowGraphData<SlotName> g_data = get_kwarg_dataflow_graph_data(g);

  std::unordered_set<Node> nodes =
      set_intersection(g_data.nodes, subgraph_nodes);

  std::unordered_set<KwargDataflowEdge<SlotName>> edges =
      filter(g_data.edges, [&](KwargDataflowEdge<SlotName> const &e) -> bool {
        return contains(subgraph_nodes, e.src.node) &&
               contains(subgraph_nodes, e.dst.node);
      });

  std::unordered_set<KwargDataflowOutput<SlotName>> outputs = filter(
      g_data.outputs, [&](KwargDataflowOutput<SlotName> const &o) -> bool {
        return contains(subgraph_nodes, o.node);
      });

  KwargDataflowGraphData<SlotName> subgraph_data =
      KwargDataflowGraphData<SlotName>{
          /*nodes=*/nodes,
          /*edges=*/edges,
          /*outputs=*/outputs,
      };

  return view_from_kwarg_dataflow_graph_data(subgraph_data);
}

} // namespace FlexFlow

#endif
