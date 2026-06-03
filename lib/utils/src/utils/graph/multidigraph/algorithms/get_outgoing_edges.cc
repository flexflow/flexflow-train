#include "utils/graph/multidigraph/algorithms/get_outgoing_edges.h"
#include "utils/containers/group_by.h"
#include "utils/containers/map_values.h"
#include "utils/containers/set_of.h"
#include "utils/graph/multidigraph/algorithms/get_edges.h"
#include "utils/graph/node/algorithms.h"
#include <unordered_set>

namespace FlexFlow {

std::unordered_set<MultiDiEdge> get_outgoing_edges(MultiDiGraphView const &g,
                                                   Node const &n) {
  MultiDiEdgeQuery query = MultiDiEdgeQuery{
      query_set<Node>::match_single_value(n),
      query_set<Node>::matchall(),
  };

  return g.query_edges(query);
}

std::unordered_map<Node, std::unordered_set<MultiDiEdge>>
    get_outgoing_edges(MultiDiGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  MultiDiEdgeQuery query = MultiDiEdgeQuery{
      query_set<Node>::match_values_in(set_of(ns)),
      query_set<Node>::matchall(),
  };

  std::unordered_map<Node, std::unordered_set<MultiDiEdge>> result = map_values(
      group_by(g.query_edges(query),
               [&](MultiDiEdge const &e) { return g.get_multidiedge_src(e); })
          .l_to_r(),
      [](nonempty_unordered_set<MultiDiEdge> const &s)
          -> std::unordered_set<MultiDiEdge> {
        return s.unwrap_as_unordered_set();
      });

  for (Node const &n : ns) {
    result[n];
  }

  return result;
}

} // namespace FlexFlow
