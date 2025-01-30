#include "utils/graph/dataflow_graph/algorithms/get_outgoing_edges.h"
#include "utils/containers/sorted_by.h"

namespace FlexFlow {

std::unordered_set<DataflowEdge> get_outgoing_edges(DataflowGraphView const &g,
                                                    Node const &n) {
  return g.query_edges(DataflowEdgeQuery{
      {n},
      query_set<int>::matchall(),
      query_set<Node>::matchall(),
      query_set<int>::matchall(),
  });
}

std::unordered_set<DataflowEdge>
    get_outgoing_edges(DataflowGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  DataflowEdgeQuery query = DataflowEdgeQuery{
      query_set<Node>{ns},
      query_set<int>::matchall(),
      query_set<Node>::matchall(),
      query_set<int>::matchall(),
  };
  return g.query_edges(query);
}

} // namespace FlexFlow
