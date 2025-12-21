#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_TRANSITIVE_REDUCED_KWARG_DATAFLOW_GRAPH_GET_TRANSITIVE_REDUCED_BOUNDARY_NODES_FOR_KWARG_DATAFLOW_GRAPH_SPLIT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_GRAPH_KWARG_DATAFLOW_GRAPH_ALGORITHMS_TRANSITIVE_REDUCED_KWARG_DATAFLOW_GRAPH_GET_TRANSITIVE_REDUCED_BOUNDARY_NODES_FOR_KWARG_DATAFLOW_GRAPH_SPLIT_H

#include "utils/graph/kwarg_dataflow_graph/algorithms/transitive_reduced_kwarg_dataflow_graph/transitive_reduced_kwarg_dataflow_graph_view.dtg.h"
#include "utils/graph/series_parallel/binary_sp_decomposition_tree/binary_series_split.dtg.h"
#include "utils/graph/dataflow_graph/algorithms/transitive_reduced_dataflow_graph/split_boundary_nodes.dtg.h"
#include "utils/graph/kwarg_dataflow_graph/algorithms/transitive_reduced_kwarg_dataflow_graph/get_transitive_reduced_kwarg_dataflow_edges_across_split.h"

namespace FlexFlow {

template <typename SlotName>
SplitBoundaryNodes get_transitive_reduced_boundary_nodes_for_kwarg_dataflow_graph_split(
    TransitiveReducedKwargDataflowGraphView<SlotName> const &tr_g, BinarySeriesSplit const &split) {

  std::unordered_set<KwargDataflowEdge<SlotName>> edges =
      get_transitive_reduced_kwarg_dataflow_edges_across_split(tr_g, split);

  std::unordered_set<Node> src_boundary_nodes =
      transform(edges, [](KwargDataflowEdge<SlotName> const &e) { return e.src.node; });

  std::unordered_set<Node> dst_boundary_nodes =
      transform(edges, [](KwargDataflowEdge<SlotName> const &e) { return e.dst.node; });

  return SplitBoundaryNodes{
      /*pre_split_boundary=*/src_boundary_nodes,
      /*post_split_boundary=*/dst_boundary_nodes,
  };
}

} // namespace FlexFlow

#endif
