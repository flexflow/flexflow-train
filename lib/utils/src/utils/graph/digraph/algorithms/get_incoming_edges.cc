#include "utils/graph/digraph/algorithms/get_incoming_edges.h"
#include "utils/containers/group_by.h"
#include "utils/containers/map_values.h"
#include "utils/containers/set_of.h"

namespace FlexFlow {

std::unordered_set<DirectedEdge> get_incoming_edges(DiGraphView const &g,
                                                    Node const &n) {
  return g.query_edges(DirectedEdgeQuery{
      query_set<Node>::matchall(),
      query_set<Node>::match_single_value(n),
  });
}

std::unordered_map<Node, std::unordered_set<DirectedEdge>>
    get_incoming_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &ns) {
  std::unordered_map<Node, std::unordered_set<DirectedEdge>> result =
      map_values(group_by(g.query_edges(DirectedEdgeQuery{
                              query_set<Node>::matchall(),
                              query_set<Node>::match_values_in(set_of(ns)),
                          }),
                          [](DirectedEdge const &e) { return e.dst; })
                     .l_to_r(),
                 [](nonempty_unordered_set<DirectedEdge> const &s)
                     -> std::unordered_set<DirectedEdge> {
                   return s.unwrap_as_unordered_set();
                 });

  for (Node const &n : ns) {
    result[n];
  }

  return result;
}

} // namespace FlexFlow
