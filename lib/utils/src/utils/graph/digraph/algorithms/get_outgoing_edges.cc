#include "utils/graph/digraph/algorithms/get_outgoing_edges.h"
#include "utils/containers/group_by.h"
#include "utils/containers/map_values.h"
#include "utils/containers/set_of.h"
#include "utils/nonempty_unordered_set/nonempty_unordered_set.h"
#include "utils/containers/unordered_map_from_map.h"

namespace FlexFlow {

std::unordered_map<Node, std::unordered_set<DirectedEdge>>
    get_outgoing_edges(DiGraphView const &g,
                       std::unordered_set<Node> const &ns) {

  std::map<Node, nonempty_set<DirectedEdge>> by_src =
     group_by(g.query_edges(DirectedEdgeQuery{
                  query_set<Node>::match_values_in(set_of(ns)),
                  query_set<Node>::matchall(),
              }),
              [](DirectedEdge const &e) { return e.src; })
         .l_to_r();

  std::map<Node, std::unordered_set<DirectedEdge>> result =
      map_values(by_src,
                 [](nonempty_set<DirectedEdge> const &s) -> std::unordered_set<DirectedEdge> {
                   return s.unwrap_as_unordered_set();
                 });

  for (Node const &n : ns) {
    result[n];
  }

  return unordered_map_from_map(result);
}

std::unordered_set<DirectedEdge> get_outgoing_edges(DiGraphView const &g,
                                                    Node const &n) {
  return g.query_edges(DirectedEdgeQuery{
      query_set<Node>::match_single_value(n),
      query_set<Node>::matchall(),
  });
}

} // namespace FlexFlow
