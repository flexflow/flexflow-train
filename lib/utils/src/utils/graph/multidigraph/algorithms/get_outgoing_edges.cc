#include "utils/graph/multidigraph/algorithms/get_outgoing_edges.h"
#include "utils/containers/group_by.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/node/algorithms.h"
#include <unordered_set>
namespace FlexFlow {

std::unordered_set<MultiDiEdge> get_outgoing_edges(MultiDiGraphView const &g,
                                                   Node const &n) {
  return g.query_edges(MultiDiEdgeQuery{{n}, query_set<Node>::matchall()});
}

std::unordered_map<Node, std::unordered_set<MultiDiEdge>>
    get_outgoing_edges(MultiDiGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  std::unordered_map<Node, std::unordered_set<MultiDiEdge>> result =
      group_by(g.query_edges(MultiDiEdgeQuery{query_set<Node>{ns},
                                              query_set<Node>::matchall()}),
               [&](MultiDiEdge const &e) { return g.get_multidiedge_src(e); });

  for (Node const &n : ns) {
    result[n];
  }

  return result;
}

} // namespace FlexFlow
