#include "utils/graph/dataflow_graph/algorithms/get_subgraph_outgoing_edges.h"
#include "utils/containers/set_minus.h"
#include "utils/containers/set_of.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

std::unordered_set<DataflowEdge>
    get_subgraph_outgoing_edges(DataflowGraphView const &g,
                                std::unordered_set<Node> const &ns) {

  std::unordered_set<Node> all_nodes = get_nodes(g);
  query_set<Node> dst_query =
      query_set<Node>::match_values_in(set_of(set_minus(all_nodes, ns)));

  DataflowEdgeQuery query = DataflowEdgeQuery{
      query_set<Node>::match_values_in(set_of(ns)),
      query_set<nonnegative_int>::matchall(),
      dst_query,
      query_set<nonnegative_int>::matchall(),
  };

  return g.query_edges(query);
}

} // namespace FlexFlow
