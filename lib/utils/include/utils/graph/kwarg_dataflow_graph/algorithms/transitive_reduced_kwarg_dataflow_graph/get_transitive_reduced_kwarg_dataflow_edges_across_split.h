#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_TRANSITIVE_REDUCED_KWARG_DATAFLOW_GRAPH_GET_TRANSITIVE_REDUCED_KWARG_DATAFLOW_EDGES_ACROSS_SPLIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_TRANSITIVE_REDUCED_KWARG_DATAFLOW_GRAPH_GET_TRANSITIVE_REDUCED_KWARG_DATAFLOW_EDGES_ACROSS_SPLIT_H

#include "utils/graph/kwarg_dataflow_graph/algorithms/transitive_reduced_kwarg_dataflow_graph/transitive_reduced_kwarg_dataflow_graph_view.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_series_split.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_sp_decomposition_tree.h"
#include "utils/graph/digraph/algorithms/get_edges_from_subgraph_to_subgraph.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/get_kwarg_dataflow_edges_from_node_to_node.h"
#include "utils/containers/flatmap.h"

namespace FlexFlow {

template <typename SlotName>
std::unordered_set<KwargDataflowEdge<SlotName>> get_transitive_reduced_kwarg_dataflow_edges_across_split(
    TransitiveReducedKwargDataflowGraphView<SlotName> const &tr_g, BinarySeriesSplit const &split) {

  std::unordered_set<Node> src_subgraph =
      unordered_set_of(get_leaves(split.get_left_child()));
  std::unordered_set<Node> dst_subgraph =
      unordered_set_of(get_leaves(split.get_right_child()));

  std::unordered_set<DirectedEdge> raw_edges =
      get_edges_from_subgraph_to_subgraph(
          tr_g.transitive_reduction, src_subgraph, dst_subgraph);

  return flatmap(raw_edges, [&](DirectedEdge const &e) {
    return get_kwarg_dataflow_edges_from_node_to_node(
        tr_g.full_kwarg_dataflow_graph, e.src, e.dst);
  });
}

} // namespace FlexFlow

#endif
