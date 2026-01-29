#include "utils/graph/series_parallel/sp_ization/node_role.h"
#include "utils/graph/algorithms.h"
#include "utils/graph/digraph/algorithms/get_predecessors.h"
#include "utils/graph/digraph/algorithms/get_successors.h"
#include "utils/graph/digraph/directed_edge.dtg.h"
#include "utils/graph/node/algorithms.h"

namespace FlexFlow {

DiGraph delete_nodes_of_given_role(
    DiGraph g,
    NodeRole const &role,
    std::unordered_map<Node, NodeRole> const &node_roles) {
  for (Node const &n : get_nodes(g)) {
    if (node_roles.at(n) == role) {
      for (Node const &pred : get_predecessors(g, n)) {
        for (Node const &succ : get_successors(g, n)) {
          g.add_edge(DirectedEdge{pred, succ});
        }
      }
      remove_node(g, n);
    }
  }
  return g;
}

} // namespace FlexFlow
