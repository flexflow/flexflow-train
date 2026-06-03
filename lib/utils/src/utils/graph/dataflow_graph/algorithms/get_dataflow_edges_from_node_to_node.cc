#include "utils/graph/dataflow_graph/algorithms/get_dataflow_edges_from_node_to_node.h"

namespace FlexFlow {

std::unordered_set<DataflowEdge> get_dataflow_edges_from_node_to_node(
    DataflowGraphView const &g, Node const &src, Node const &dst) {
  return g.query_edges(DataflowEdgeQuery{
      /*src_nodes=*/query_set<Node>::match_single_value(src),
      /*src_idxs=*/query_set<nonnegative_int>::matchall(),
      /*dst_nodes=*/query_set<Node>::match_single_value(dst),
      /*dst_idxs=*/query_set<nonnegative_int>::matchall(),
  });
}

} // namespace FlexFlow
