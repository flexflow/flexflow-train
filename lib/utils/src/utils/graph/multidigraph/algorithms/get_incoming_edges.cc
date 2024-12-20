#include "utils/graph/multidigraph/algorithms/get_incoming_edges.h"
#include "utils/containers/group_by.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/multidigraph/multidiedge.dtg.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

std::unordered_set<MultiDiEdge> get_incoming_edges(MultiDiGraphView const &g,
                                                   Node const &n) {
  return g.query_edges(MultiDiEdgeQuery{query_set<Node>::matchall(), {n}});
}

std::unordered_map<Node, std::unordered_set<MultiDiEdge>>
    get_incoming_edges(MultiDiGraphView const &g) {
  std::unordered_map<Node, std::unordered_set<MultiDiEdge>> result =
      group_by(get_edges(g),
               [&](MultiDiEdge const &e) { return g.get_multidiedge_dst(e); });

  for (Node const &n : get_nodes(g)) {
    result[n];
  }

  return result;
}

} // namespace FlexFlow
